import cv2
import numpy as np


class ClosestObjectDetector:
    """
    Iterates a depth image in layers beginning at depth 0.

    In fact, this is not a good hand detector.
    The idea of this detector might be useful for other applications.
    """

    def __init__(self, min_depth=100, max_depth=1200, steps=22, min_contour_area=200, cube=200):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.steps = steps
        self.min_contour_area = min_contour_area
        self.cube = cube
        self.depth_per_step = (max_depth - min_depth) / steps

    def detect(self, image):
        print(np.mean(image[image != 0]))
        for i in range(self.steps):
            min_bound = i * self.depth_per_step + self.min_depth
            max_bound = (i + 1) * self.depth_per_step + self.min_depth

            image_layer = image.copy()
            image_layer[image_layer < min_bound] = 0
            image_layer[image_layer > max_bound] = 0

            # Set all nonzero pixels to to the range 1-255
            image_layer[image_layer != 0] = 100

            ret, thresh = cv2.threshold(image_layer, 1, 255, cv2.THRESH_BINARY)
            # Convert dtype to uint8
            thresh = cv2.convertScaleAbs(thresh)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour_idx in range(len(contours)):
                if cv2.contourArea(contours[contour_idx]) > self.min_contour_area:
                    bbox = self.find_hand_bbox(contours[contour_idx], image.shape)
                    return bbox

    def find_hand_bbox(self, contour, image_shape):
        M = cv2.moments(contour)
        cx = int(np.rint(M["m10"] / M["m00"]))
        cy = int(np.rint(M["m01"] / M["m00"]))

        half_cube = self.cube / 2
        xstart = int(max(cx - half_cube, 0))
        xend = int(min(cx + half_cube, image_shape[1] - 1))
        ystart = int(max(cy - half_cube, 0))
        yend = int(min(cy + half_cube, image_shape[0] - 1))

        return np.array([xstart, ystart, xend, yend])
