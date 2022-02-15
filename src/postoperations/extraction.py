"""
Finger extraction

Extracts fingers by interpolation keypoints with lines and
removing the palm.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from src.utils.debugging import timing
from src.utils.paths import OTHER_DIR
from src.utils.plots import _plot_image_with_skeleton, save_show_fig

joint_indices = {'mcp': np.arange(1, 21, 4),
                 'pip': np.arange(2, 21, 4),
                 'dip': np.arange(3, 21, 4),
                 'tip': np.arange(4, 21, 4)}


@timing
def interpolate_curve(ax, image, joints2d, joint_type):
    dips = joints2d[joint_indices[joint_type]]
    # dips = dips[dips[..., 0].argsort()]

    # ax.scatter(dips[..., 0], dips[..., 1], c='b', marker='o', s=50, alpha=1)

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(dips, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # Interpolation
    interpolator = interpolate.interp1d(distance, dips, kind='slinear', axis=0, fill_value="extrapolate")

    delta = 0.2
    alpha = np.linspace(0 - delta, 1 + delta, 75)
    interpolation = interpolator(alpha)
    interpolation[interpolation < 0] = 0
    interpolation[interpolation >= image.shape[0]] = image.shape[0]
    ax.plot(*interpolation.T, '-')
    return interpolation


def display_interpolated_image(image, joints2d, show_fig=True, fig_location=None, figsize=(4, 3)):
    fig, ax = plt.subplots(1, figsize=figsize)
    _plot_image_with_skeleton(fig, ax, image, joints2d)
    for joint_type in joint_indices:
        interpolate_curve(ax, image, joints2d, joint_type)
    save_show_fig(fig, fig_location, show_fig)


def line_rect_intersection(point1, point2, rect):
    Ax = float(point1[0])
    Ay = float(point1[1])
    Bx = float(point2[0])
    By = float(point2[1])
    w = float(rect[2])
    h = float(rect[3])
    s = (Ay - By) / (Ax - Bx)
    x, y = 0, 0
    if -h / 2 <= s * w / 2 <= h / 2:
        if Ax > Bx:
            # the right edge
            x = 0
        if Ax < Bx:
            # the left edge
            x = w
        y = s * (x - Ax) + Ay
    if -w / 2 <= (h / 2) / s <= w / 2:
        if Ay > By:
            # the top edge
            y = 0
        if Ay < By:
            # the bottom edge
            y = h
        x = (y + s * Ax - Ay) / s
    return [int(x), int(y)]


@timing
def draw_interpolated_lines(image, joints):
    for id_previous, rhs in enumerate(joints[2:-1], 1):
        lhs = joints[id_previous]
        cv.line(image, (lhs[0], lhs[1]), (rhs[0], rhs[1]), (0, 0, 0))
    # From index finger to thumb
    index = joints[1]
    thumb = joints[0]
    intersection = line_rect_intersection(index, thumb, (0, 0, 256, 256))
    cv.line(image, (index[0], index[1]), (intersection[0], intersection[1]), (0, 0, 0))

    # From a ring finger to a little finger
    ring = joints[-2]
    little = joints[-1]
    intersection = line_rect_intersection(ring, little, (0, 0, 256, 256))
    cv.line(image, (ring[0], ring[1]), (intersection[0], intersection[1]), (0, 0, 0))

    return image


class FingerExtractor:

    def __init__(self):
        pass

    @timing
    def extract_hulls(self, depth_image, joints2d):
        # the interesting thing is that range of pixels is [-1, 1]
        # with hand probably in negative numbers if there is some background
        ret, thresh = cv.threshold(depth_image, 0.2, 1, cv.THRESH_BINARY_INV)
        thresh = cv.convertScaleAbs(thresh)
        hulls = []

        for joint_type in ['pip', 'dip', 'tip']:
            # Copy thresholded image
            mask = thresh.copy()
            # Separate palm from fingers
            joints_of_same_type = joints2d[joint_indices[joint_type]]
            mask = draw_interpolated_lines(mask, joints_of_same_type)
            seed_point = joints2d[joint_indices['mcp'][3]]
            cv.floodFill(mask, None, (seed_point[0], seed_point[1]), 0)

            # Check there are 5 big convex hulls by finding contours
            contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            big_enough_contours = []
            for contour in contours:
                area = cv.contourArea(contour)
                if area > 100:
                    big_enough_contours.append(contour)

            if len(big_enough_contours) == 5:
                # Draw convex hulls into the original image
                for contour in big_enough_contours:
                    hull = cv.convexHull(contour, returnPoints=True)
                    hulls.append(hull)
                break
        return hulls


class ExtractedFingersDisplay:

    def __init__(self):
        self.window_name = 'fingercrops'
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)

    def display(self, color_image, hulls):
        for hull in hulls:
            cv.drawContours(color_image, [hull], -1, (0, 0, 200), 2)
        cv.imshow(self.window_name, color_image)
        cv.waitKey(0)

    def __del__(self):
        cv.destroyWindow(self.window_name)


if __name__ == "__main__":
    # datetime = F"20220201-172311"  # not so perfect, requires correction
    datetime = F"20220201-172322"  # perfect for correction
    # datetime = F"20220201-172325"  # rotated hand, what happens to correction?
    # datetime = F"20220201-172328"  # rotated hand, what happens to correction?
    datetime = F"20220201-172312"  # nice pose
    img_path = OTHER_DIR.joinpath(F"extraction/{datetime}_image.npy")
    jnt_path = OTHER_DIR.joinpath(F"extraction/{datetime}_joints.npy")

    img = np.load(img_path)
    jnt = np.load(jnt_path)
    jnt2d = jnt[:, :2] * 255

    extractor = FingerExtractor()
    display = ExtractedFingersDisplay()

    hulls = extractor.extract_hulls(img, jnt2d)
    color_image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    display.display(color_image, hulls)

    # display_removed_palm(img, jnt2d)
    # display_interpolated_image(img, jnt2d)
