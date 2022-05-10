from abc import ABC, abstractmethod

import numpy as np
from acceptance.gesture_acceptance_result import GestureAcceptanceResult


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
    def recognize(self, keypoints_xyz: np.ndarray) -> GestureAcceptanceResult:
        pass
