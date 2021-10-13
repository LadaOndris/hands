import tensorflow as tf
import tensorflow_addons as tfa

from src.estimation.blazepose.data.rotate import rotate_tensor, rotation_angle_from_21_keypoints
from src.estimation.jgrp2o.preprocessing import extract_bboxes, get_resize_coeffs, resize_coords
from src.estimation.jgrp2o.preprocessing_com import ComPreprocessor, crop_to_bcube, crop_to_bounding_box
from src.utils.camera import Camera
from src.utils.imaging import normalize_to_range, resize_bilinear_nearest


def preprocess(image, joints, camera: Camera, heatmap_sigma: int, cube_size=200,
               image_target_size=256, output_target_size=64):
    """
    Prepares data for BlazePose training.

    Crops the image, resizes it to the [256, 256, 1] range and rotates it,
    so that the palm is always at the bottom of the image, which should
    decrease the required complexity of the model.
    Generates heatmaps of shape [64, 64, n_joints].
    Converts XYZ joint coordinates to UVZ coordinates and normalizes the UV coordinates
    and Z coordinates to the [0, 1] and [-1, 1] range respectively.
    Appends hand presence and handedness flags to joint coordinates.

    Parameters
    ----------
    image A depth image of shape [256, 256, 1]
    joints  Joint coordinates of shape [n_joints, 3] in XYZ mode.
    camera  A camera instance with specific parameters that captured the given image.

    Returns
    -------
    image   Image of shape [256, 256, 1]
    joints  UVZ joint coords + hand presence + handedness with shape [n_joints, 5]
    heatmap Generated heatmaps for each joint with shape [64, 64, n_joints]
    """

    tf.assert_rank(image, 3)
    tf.assert_rank(joints, 2)

    uv_global = camera.world_to_pixel(joints)[..., :2]
    image = tf.cast(image, tf.float32)

    cropped_image, cropped_uv, bbox = crop_image_and_joints(image, uv_global, camera, cube_size)
    rotated_image, rotated_uv = rotate_image_and_joints(cropped_image, cropped_uv)
    resized_image, resized_uv = resize_image_and_joints(rotated_image, rotated_uv, image_target_size, bbox)
    normalized_image, normalized_uvz = normalize_image_and_joints(resized_image, resized_uv, joints[..., 2:3])

    heatmaps = generate_heatmaps(resized_uv,
                                 orig_size=tf.shape(resized_image)[:2],
                                 target_size=[output_target_size, output_target_size],
                                 sigma=heatmap_sigma)

    return normalized_image, normalized_uvz, heatmaps


def crop_image_and_joints(image, uv_coords, camera, cube_size):
    com_preprocessor = ComPreprocessor(camera, thresholding=False)
    bbox_raw = extract_bboxes(uv_coords[tf.newaxis, ...])[0]
    cropped_image = crop_to_bounding_box(image, bbox_raw)
    com = com_preprocessor.compute_coms(cropped_image[tf.newaxis, ...], offsets=bbox_raw[tf.newaxis, ..., :2])[0]
    bcube = com_preprocessor.com_to_bcube(com, size=[cube_size, cube_size, cube_size])
    bbox = cube_to_box(bcube)
    cropped_image = crop_to_bcube(image, bcube)
    # The joints can still overflow the bounding box even after the crop
    cropped_joints_uv = uv_coords - tf.cast(bbox[tf.newaxis, :2], dtype=tf.float32)
    return cropped_image, cropped_joints_uv, bbox


def rotate_image_and_joints(image, uv_coords):
    min_value = tf.reduce_min(image)
    rotation_angle = rotation_angle_from_21_keypoints(uv_coords)
    image_rotated = tfa.image.transform_ops.rotate(image, rotation_angle, fill_value=min_value)
    image_center = [tf.shape(image)[1] / 2, tf.shape(image)[0] / 2]
    uv_rotated = rotate_tensor(uv_coords, rotation_angle, center=image_center)
    return image_rotated, uv_rotated


def resize_image_and_joints(image, uv_coords, image_target_size, bbox):
    # Resize image
    resized_image = resize_bilinear_nearest(image, [image_target_size, image_target_size])
    # Resize coordinates
    resize_coeffs = get_resize_coeffs(bbox, target_size=[image_target_size, image_target_size])
    resized_joints_uv = resize_coords(uv_coords, resize_coeffs)
    return resized_image, resized_joints_uv


def normalize_image_and_joints(image, uv_coords, z_coords):
    min_value = tf.reduce_min(image)
    max_value = tf.reduce_max(image)
    normalized_image = normalize_to_range(image, [-1, 1], min_value, max_value)
    normalized_z = normalize_to_range(z_coords, [-1, 1], min_value, max_value)
    normalized_uv = normalize_to_range(uv_coords, [0, 1], 0, 255)
    normalized_joints_uvz = tf.concat([normalized_uv, normalized_z], axis=-1)
    return normalized_image, normalized_joints_uvz


def cube_to_box(cube):
    """
    Exludes a cube's z axis, transforming it into a box.

    Parameters
    ----------
    cube    A cube defined as [x_start, y_start, z_start, x_end, y_end, z_end]

    Returns
    -------
    box     A box defined as [x_start, y_start, x_end, y_end]
    """
    return tf.concat([cube[..., 0:2], cube[..., 3:5]], axis=-1)


@tf.function
def generate_heatmaps(keypoints, orig_size, target_size, sigma):
    """

    Parameters
    ----------
    keypoints   A tensor of keypoints of shape [n_points, 2]
    orig_size   Maximum x and y coordinates of the given keypoints, e.g. [480, 640].
    target_size Size of the heatmap, e.g. [64, 64].

    Returns
    -------

    """

    heatmap_size = tf.convert_to_tensor([target_size[0], target_size[1], 1], dtype=tf.int32)
    image_size = tf.convert_to_tensor([orig_size[0], orig_size[1], 1], dtype=tf.float32)
    scale = tf.cast(heatmap_size[tf.newaxis, :2], tf.float32) / image_size[tf.newaxis, :2]

    keypoints = keypoints * scale
    num_keypoints = tf.shape(keypoints)[0]
    heatmaps = tf.TensorArray(dtype=tf.float32, size=num_keypoints)
    for i in range(num_keypoints):
        heatmap = tf.zeros(heatmap_size)
        heatmap = draw_gaussian_point(heatmap, keypoints[i], sigma=sigma)
        heatmaps = heatmaps.write(i, heatmap)
    return heatmaps.stack()


@tf.function
def draw_gaussian_point(image, point, sigma):
    """
    Draw a 2D gaussian.

    Adapted from https://github.com/princeton-vl/pose-hg-train/blob/master/src/pypose/draw.py.

    Parameters
    ----------
    image   Input image of shape [height, width, 1]
    point   Point in format [x, y]
    sigma   Sigma param in Gaussian

    Returns
    -------
    updated_image  An image of shape [height, width, 1] with a gaussian point drawn in it.
    """
    tf.assert_rank(image, 3)
    tf.assert_rank(point, 1)
    tf.assert_rank(sigma, 0)

    # Check that any part of the gaussian is in-bounds
    ul = [int(point[0] - 3 * sigma), int(point[1] - 3 * sigma)]
    br = [int(point[0] + 3 * sigma + 1), int(point[1] + 3 * sigma + 1)]
    if (ul[0] > image.shape[1] or ul[1] >= image.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return image

    # Generate gaussian
    size = 6 * sigma + 1
    x = tf.range(0, size, dtype=tf.float32)
    y = x[:, tf.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = tf.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = tf.maximum(0, -ul[0]), tf.minimum(br[0], image.shape[1]) - ul[0]
    g_y = tf.maximum(0, -ul[1]), tf.minimum(br[1], image.shape[0]) - ul[1]
    # Image range
    img_x = tf.maximum(0, ul[0]), tf.minimum(br[0], image.shape[1])
    img_y = tf.maximum(0, ul[1]), tf.minimum(br[1], image.shape[0])

    top = image[:img_y[0], ...]
    middle = image[img_y[0]:img_y[1], ...]
    bottom = image[img_y[1]:, ...]

    left = middle[:, :img_x[0]]
    right = middle[:, img_x[1]:]
    center = g[g_y[0]:g_y[1], g_x[0]:g_x[1], tf.newaxis]

    updated_middle = tf.concat([left, center, right], axis=1)
    image = tf.concat([top, updated_middle, bottom], axis=0)
    return image


def try_dataset_preprocessing():
    from src.datasets.bighand.dataset import BighandDataset, BIGHAND_DATASET_DIR
    from src.utils.plots import plot_image_with_skeleton

    prepare_fn = lambda image, joints: preprocess(image, joints, Camera('bighand'), heatmap_sigma=4, cube_size=180)
    prepare_fn_shape = (tf.TensorShape([256, 256, 1]), tf.TensorShape([21, 3]), tf.TensorShape([64, 64, 21]))
    ds = BighandDataset(BIGHAND_DATASET_DIR, batch_size=1, shuffle=False,
                        prepare_output_fn=prepare_fn, prepare_output_fn_shape=prepare_fn_shape)
    train_iterator = iter(ds.train_dataset)

    for image, joints, heatmap in train_iterator:
        plot_image_with_skeleton(image, joints[:, :2])
        break


def try_preprocessing():
    from src.datasets.bighand.dataset import BighandDataset, BIGHAND_DATASET_DIR
    from src.utils.plots import plot_image_with_skeleton

    ds = BighandDataset(BIGHAND_DATASET_DIR, batch_size=1, shuffle=True)
    train_iterator = iter(ds.train_dataset)

    for image, joints in train_iterator:
        image, joints, heatmaps = preprocess(image[0], joints[0], Camera('bighand'), heatmap_sigma=4, cube_size=180)
        plot_image_with_skeleton(image, joints[:, :2])
        break


if __name__ == "__main__":
    try_preprocessing()
