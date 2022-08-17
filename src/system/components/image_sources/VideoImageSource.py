import os

from cv2 import cv2

from src.system.components.base import ImageSource


class VideoImageSource(ImageSource):

    def __init__(self, file_path: str):
        self._check_file_exists(file_path)
        self.cap = cv2.VideoCapture(file_path)
        self.previous_image = None

    def _check_file_exists(self, file_path: str):
        if not os.path.isfile(file_path):
            raise RuntimeError(f'File {file_path} does not exist.')

    def get_new_image(self):
        if self.cap.isOpened():
            success, image = self.cap.read()
            if success:
                self.previous_image = image
                return image
            else:
                raise RuntimeError("No more video frames.")
        raise RuntimeError("VideoCapture is not opened.")

    def get_previous_image(self):
        return self.previous_image

    def __exit__(self):
        self.cap.release()
