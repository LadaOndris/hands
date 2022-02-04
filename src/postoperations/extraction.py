"""
Finger extraction
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


def display_image(image, joints2d, show_fig=True, fig_location=None, figsize=(4, 3)):
    fig, ax = plt.subplots(1, figsize=figsize)

    _plot_image_with_skeleton(fig, ax, image, joints2d)

    for joint_type in joint_indices:
        interpolate_curve(ax, image, joints2d, joint_type)

    save_show_fig(fig, fig_location, show_fig)


@timing
def find_convex_hull(image):
    ret, thresh = cv.threshold(image, 0.7, 1, cv.THRESH_BINARY_INV)

    thresh = cv.convertScaleAbs(thresh)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    biggest_contour = max(contours, key=lambda x: cv.contourArea(x))
    hull = cv.convexHull(biggest_contour, returnPoints=False)
    defects = cv.convexityDefects(biggest_contour, hull)
    return thresh, biggest_contour, hull, defects


def display_contour(image):
    thresh, biggest_contour, hull, defects = find_convex_hull(image)
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    cnt = biggest_contour

    cv.drawContours(drawing, [biggest_contour], -1, (0, 255, 10), 1)
    defects = defects[defects[..., 3].argsort(axis=0).squeeze()][::-1]
    for i in range(5):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        # cv.line(drawing, start, end, [0, 255, 0], 2)
        # cv.circle(drawing, far, 5, [0, 0, 255], -1)
        cv.circle(drawing, start, 3, [255, 0, 0], 2)

    # cv.drawContours(drawing, [hull], -1, (255, 0, 0), 2)

    plt.imshow(drawing)  # , 'gray', vmin=0, vmax=1)
    plt.show()
    # cv.imshow("hull", img)


# datetime = F"20220201-172311" # not so perfect, requires correction
# datetime = F"20220201-172322"  # perfect for correction
# datetime = F"20220201-172325"  # rotated hand, what happens to correction?
datetime = F"20220201-172328"  # rotated hand, what happens to correction?
img_path = OTHER_DIR.joinpath(F"extraction/{datetime}_image.npy")
jnt_path = OTHER_DIR.joinpath(F"extraction/{datetime}_joints.npy")

img = np.load(img_path)
jnt = np.load(jnt_path)
jnt2d = jnt[:, :2] * 255

display_contour(img)
# display_image(img, jnt2d)
