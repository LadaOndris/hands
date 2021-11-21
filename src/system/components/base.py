from abc import ABC, abstractmethod


class ImageSource(ABC):

    @abstractmethod
    def next_image(self):
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
    def update(self, keypoints, image):
        pass
