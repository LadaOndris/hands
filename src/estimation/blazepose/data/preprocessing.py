import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src.estimation.jgrp2o.preprocessing import extract_bboxes
from src.estimation.jgrp2o.preprocessing_com import ComPreprocessor, crop_to_bounding_box
from src.utils.camera import Camera
from src.utils.imaging import normalize_to_range, resize_bilinear_nearest


def preprocess(image, joints, camera: Camera, heatmap_sigma: int, cube_size=200):
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

    com_preprocessor = ComPreprocessor(camera)

    uv_global = camera.world_to_pixel(joints)[..., :2]
    image = tf.cast(image, tf.float32)

    rotation_angle = determine_rotation_angle(uv_global)
    image_rotated = tfa.image.transform_ops.rotate(image, rotation_angle)
    uv_global_rotated = rotate_joints(uv_global)

    bbox_raw = extract_bboxes(uv_global_rotated)
    cropped_image = crop_to_bounding_box(image_rotated, bbox_raw)
    coms = com_preprocessor.compute_coms(cropped_image, offsets=bbox_raw[..., :2])
    bcube = com_preprocessor.com_to_bcube(coms, size=cube_size)
    cropped_image = crop_to_bounding_box(image_rotated, cube_to_box(bcube))

    resized_image = resize_bilinear_nearest(cropped_image, [256, 256])
    normalized_image = normalize_to_range(resized_image, range=[-1, 1])

    heatmaps = generate_heatmaps(uv_global, orig_size=tf.shape(image)[:2], target_size=[64, 64])

    return normalized_image, joints, heatmaps


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
    return tf.concat([cube[..., 0:2], cube[..., 3:5]])


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

    heatmap_size = np.array([target_size[0], target_size[1], 1])
    image_size = np.array([orig_size[0], orig_size[1], 1])

    keypoints = keypoints * heatmap_size[np.newaxis, :2] / image_size[np.newaxis, :2]
    heatmaps = []
    for keypoint in keypoints:
        heatmap = tf.zeros(heatmap_size)
        heatmap = draw_gaussian_point(heatmap, keypoint, sigma=sigma)
        heatmaps.append(heatmap)
    return tf.concat(heatmaps, axis=-1)


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