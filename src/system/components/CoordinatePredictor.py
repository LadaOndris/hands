from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from src.system.components.base import Detector, Estimator
from src.utils.camera import Camera


class CoordinatePrediction:

    def __init__(self, world_coordinates: np.ndarray, image_coordinates: np.ndarray):
        self.world_coordinates = world_coordinates
        self.image_coordinates = image_coordinates


class CoordinatePredictor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> CoordinatePrediction:
        pass


class CustomCoordinatePredictor(CoordinatePredictor):

    def __init__(self, detector: Detector, estimator: Estimator, camera: Camera):
        super().__init__()
        self.detector = detector
        self.estimator = estimator
        self.camera = camera

    def predict(self, image: np.ndarray) -> CoordinatePrediction:
        image = tf.convert_to_tensor(image)

        rectangle = self.detector.detect(image)[0]
        if not tf.experimental.numpy.allclose(rectangle, 0):
            hand_presence_flag, keypoints_uvz, normalized_keypoints_uvz, normalized_image, crop_offset_uv = \
                self.estimator.estimate(image, rectangle)
            if hand_presence_flag:
                keypoints_xyz = self.camera.pixel_to_world(keypoints_uvz)
                return CoordinatePrediction(keypoints_xyz, keypoints_uvz)
        return None


class MediapipeCoordinatePredictor(CoordinatePredictor):

    def __init__(self):
        super().__init__()
