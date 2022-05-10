from typing import Tuple

import cv2
import numpy as np

from src.system.components.base import Display
from src.system.components.image_source import RealSenseCameraWrapper


class EmptyDisplay(Display):

    def update(self, image, keypoints=None, bounding_boxes=None, gesture_label: str = None):
        pass


class OpencvDisplay(Display):

    def __init__(self):
        self.window_same = 'window'
        cv2.namedWindow(self.window_same, cv2.WINDOW_NORMAL)

    def update(self, image, keypoints=None, bounding_boxes=None, gesture_label: str = None):
        # Convert depth image to something cv2 can understand
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.4), cv2.COLORMAP_BONE)

        # Draw rectangles in the depth colormap image
        if bounding_boxes is not None:
            for rect in bounding_boxes:
                cv2.rectangle(depth_colormap, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
        if keypoints is not None:
            for point in keypoints:
                cv2.circle(depth_colormap, (point[0], point[1]), radius=0, color=(0, 0, 255), thickness=4)

        # TODO: Do not overwrite image variable.
        image = depth_colormap
        img_width = image.shape[1]

        """
        The following code is adopted from HansHirse's answer on StackOverflow.
        https://stackoverflow.com/a/56472488/3961841
        """
        # Initialize blank mask image of same dimensions for drawing the shapes
        shapes = np.zeros_like(image, np.uint8)
        # Draw shapes
        bar_height = 35
        color = self.get_color(gesture_label is not None)
        cv2.rectangle(shapes, (0, 0), (img_width, bar_height), color, cv2.FILLED)
        # Generate output by blending image with shapes image, using the shapes
        # images also as mask to limit the blending to those parts
        alpha = 0.5
        mask = shapes.astype(bool)
        image[mask] = cv2.addWeighted(image, alpha, shapes, 1 - alpha, 0)[mask]

        if gesture_label is None:
            gesture_label = "-"
        org = (int(img_width / 2), bar_height - 8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (240, 240, 240)
        thickness = 2
        image = cv2.putText(image, gesture_label, org, font, font_scale,
                            color, thickness, cv2.LINE_AA, False)

        cv2.imshow(self.window_same, image)
        # Don't wait for the user to press a key
        cv2.waitKey(1)

    def get_color(self, exists_label: bool) -> Tuple[int, int, int]:
        green = (33, 173, 38)
        blue = (181, 113, 76)

        if exists_label:
            return green
        else:
            return blue


if __name__ == "__main__":
    display = OpencvDisplay()
    realsense_wrapper = RealSenseCameraWrapper(enable_depth=True, enable_color=False)
    img_source = realsense_wrapper.get_depth_image_source()
    while True:
        img = img_source.get_new_image()
        display.update(img)
