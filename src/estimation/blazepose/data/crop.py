import tensorflow as tf

from src.estimation.jgrp2o.preprocessing import extract_bboxes
from src.estimation.jgrp2o.preprocessing_com import ComPreprocessor, crop_to_bounding_box
from src.utils.camera import Camera


class CropType:
    JOINTS_MEAN = 0
    CENTER_OF_MASS = 1
    RANDOM = 2


def get_crop_type(use_center_of_joints: bool, generate_random_crop_prob: float) -> int:
    should_generate_random_val = tf.random.uniform(shape=[1], minval=0, maxval=1)
    if should_generate_random_val < generate_random_crop_prob:
        return CropType.RANDOM

    if use_center_of_joints:
        return CropType.JOINTS_MEAN
    else:
        return CropType.CENTER_OF_MASS


def get_crop_center_point(crop_type: int, image, keypoints_uv, keypoints_xyz, camera: Camera):
    com_preprocessor = ComPreprocessor(camera, thresholding=False)

    if crop_type == CropType.CENTER_OF_MASS:
        bbox_raw = extract_bboxes(keypoints_uv[tf.newaxis, ...])[0]
        cropped_image = crop_to_bounding_box(image, bbox_raw)
        center_point_uvz = com_preprocessor.compute_coms(cropped_image[tf.newaxis, ...],
                                                         offsets=bbox_raw[tf.newaxis, ..., :2])[0]
    elif crop_type == CropType.JOINTS_MEAN:
        com_xyz = tf.reduce_mean(keypoints_xyz, axis=-2)
        center_point_uvz = camera.world_to_pixel_1d(com_xyz)
    else:  # Random crop
        random_box_center_uv = get_random_box_center(image)
        z = image[random_box_center_uv[0], random_box_center_uv[1]]
        random_box_center_uv = tf.cast(random_box_center_uv, tf.float32)
        center_point_uvz = tf.concat([random_box_center_uv, z], axis=-1)

    return center_point_uvz


def get_random_box_center(image):
    indices = tf.where(tf.squeeze(image) != 0)
    indices = tf.cast(indices, tf.int32)
    last_item_index = tf.shape(indices)[0] - 1
    sample_index = tf.random.uniform(shape=[1], minval=0, maxval=last_item_index, dtype=tf.int32)
    return indices[sample_index[0]]
