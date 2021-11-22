import json

import tensorflow as tf

from src.estimation.blazepose.data.preprocessing import cube_to_box
from src.estimation.jgrp2o.preprocessing import get_resize_coeffs
from src.estimation.blazepose.models.ModelCreator import ModelCreator
from src.estimation.jgrp2o.preprocessing_com import ComPreprocessor, crop_to_bcube, crop_to_bounding_box
from src.system.components.base import Estimator
from src.utils.camera import Camera
from src.utils.imaging import normalize_to_range, resize_bilinear_nearest
from src.utils.paths import ROOT_DIR, SRC_DIR


class BlazeposeEstimator(Estimator):

    def __init__(self, camera: Camera):
        config_path = SRC_DIR.joinpath('estimation/blazepose/configs/config_blazepose.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_config = config['model']
        test_config = config["test"]
        cube_size = model_config['cube_size']
        self.model_image_size = model_config['im_width']
        self.cube_dims = [cube_size, cube_size, cube_size]
        self.model = ModelCreator.create_model(model_config["model_type"],
                                               model_config["num_keypoints"],
                                               model_config['num_keypoint_features'])
        print("Loading model weights: " + test_config["weights_path"])
        weights_path = ROOT_DIR.joinpath(test_config["weights_path"])
        self.model.load_weights(weights_path)
        self.com_preprocessor = ComPreprocessor(camera, thresholding=False)

    @tf.function
    def estimate(self, image, rectangle):
        # original image has shape [480, 480, 1]
        normalized_images_batch, crop_offset_uv = self.preprocess(image, rectangle)
        # normalized image has shape [255, 255, 1]
        pred_joints_batch, pred_heatmap_batch, pred_presence_batch = self.model(normalized_images_batch)
        pred_joints_uvz = pred_joints_batch[0]  # coordinates are in range [0, 1]
        pred_heatmap = pred_heatmap_batch[0]  # heatmap values are in range [0, 1]
        pred_presence = pred_presence_batch[0]  # presence are probabilities for each joint in range [0, 1]
        present_joints = tf.reduce_sum(tf.cast(pred_presence > 0.5, tf.int32))
        tf.print("Present joints: ", present_joints)
        hand_presence = present_joints >= 10
        joints_uvz = self.postprocess(pred_joints_uvz, crop_offset_uv)
        return hand_presence, joints_uvz, normalized_images_batch[0]

    def preprocess(self, image, rectangle):
        """
        Parameters
        ----------
        image   Input shape is [480, 480, 1].
        rectangle A boundary around the hand with values in range [0, 480].

        Returns
        -------
        normalized_image
        """
        tf.assert_rank(image, 3)
        tf.assert_rank(rectangle, 1)
        rectangle = tf.cast(rectangle, tf.int32)
        image = tf.cast(image, tf.float32)
        # square detection box (just find center of the rectangle)
        # crop_center_point = (rectangle[:2] + rectangle[2:]) / 2
        cropped_image = crop_to_bounding_box(image, rectangle)
        center_point_uvz = self.com_preprocessor.compute_coms(cropped_image[tf.newaxis, ...],
                                                              offsets=rectangle[tf.newaxis, ..., :2])[0]
        # crop around center
        cropped_image, bbox, min_depth, max_depth = self._crop_image(image, center_point_uvz)
        # TODO: Rotate image
        # resize image to model's expected size [255, 255]
        resized_image = resize_bilinear_nearest(cropped_image, [self.model_image_size, self.model_image_size])
        # normalize depth to the [-1, 1] range
        normalized_image = normalize_to_range(resized_image, [-1, 1], min_depth, max_depth)
        return normalized_image[tf.newaxis, ...], bbox

    def _crop_image(self, image, center_point_vu):
        bcube = self.com_preprocessor.com_to_bcube(center_point_vu, size=self.cube_dims)
        bbox = cube_to_box(bcube)
        cropped_image = crop_to_bcube(image, bcube)
        min_z = bcube[2]
        max_z = bcube[5]
        return cropped_image, bbox, min_z, max_z

    def postprocess(self, pred_joints_uvz, bbox):
        resize_coeffs = get_resize_coeffs(bbox, target_size=[self.model_image_size, self.model_image_size])
        pred_joints_uv = pred_joints_uvz[..., :2]
        joints_uv = pred_joints_uv * self.model_image_size / resize_coeffs[tf.newaxis, :] + tf.cast(bbox[tf.newaxis, :2], tf.float32)
        return tf.concat([joints_uv, pred_joints_uvz[..., 2:3]], axis=-1)
