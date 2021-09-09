import tensorflow as tf
import tensorflow_addons as tfa

from src.estimation.jgrp2o.preprocessing import extract_bboxes
from src.estimation.jgrp2o.preprocessing_com import ComPreprocessor, crop_to_bounding_box
from src.utils.camera import Camera
from src.utils.imaging import normalize_to_range, resize_bilinear_nearest


def preprocess(image, joints, camera: Camera, cube_size=200):
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

    heatmaps = generate_heatmaps(uv_global, orig_size=)

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

def generate_heatmaps(keypoints, orig_size, target_size):
    """

    Parameters
    ----------
    keypoints
    orig_size   Maximum x and y coordinates of the given keypoints, e.g. [480, 640].
    target_size Size of the heatmap, e.g. [64, 64].

    Returns
    -------

    """
    pass