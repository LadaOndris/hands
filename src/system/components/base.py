from abc import ABC, abstractmethod

import numpy as np

from gestures.gesture_acceptance_result import GestureRecognitionResult


class ImageSource(ABC):

    @abstractmethod
    def get_new_image(self):
        pass

    @abstractmethod
    def get_previous_image(self):
        pass


class Detector(ABC):

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @abstractmethod
    def detect(self, image):
        pass


class Estimator(ABC):

    @abstractmethod
    def estimate(self, image, rectangle):
        pass


class KeypointsToRectangle(ABC):

    @abstractmethod
    def convert(self, keypoints):
        pass


class Display(ABC):

    @abstractmethod
    def update(self, image, keypoints=None, bounding_boxes=None, gesture_label: str = None):
        pass


class GestureRecognizer(ABC):

    @abstractmethod
    def recognize(self, keypoints_xyz: np.ndarray) -> GestureRecognitionResult:
        pass


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