import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from src.utils.debugging import timing
from src.utils.plots import plot_depth_image


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
    # kernel = np.ones((8, 8), np.uint8)
    # image = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)

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
        # cv.circle(drawing, start, 3, [255, 0, 0], 2)
        cv.line(drawing, start, far, [0, 255, 0], 2)

    # cv.drawContours(drawing, [hull], -1, (255, 0, 0), 2)

    plot_depth_image(image)
    plt.imshow(drawing)  # , 'gray', vmin=0, vmax=1)
    plt.show()
    # cv.imshow("hull", img)
