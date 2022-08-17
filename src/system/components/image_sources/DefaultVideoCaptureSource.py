import cv2

from src.system.components.base import ImageSource


class DefaultVideoCaptureSource(ImageSource):

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.previous_image = None

    def get_new_image(self):
        success, image = self.cap.read()
        self.previous_image = image
        return image

    def get_previous_image(self):
        return self.previous_image

    def __del__(self):
        self.cap.release()
