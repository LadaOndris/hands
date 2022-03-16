from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from src.utils.imaging import create_coord_pairs_np

from system.components.image_source import RealSenseCameraWrapper


class CalibratorRGBD:

    def __init__(self, depth_intrinsics, color_intrinsics):
        self.depth_intrinsics = depth_intrinsics
        self.color_intrinsics = color_intrinsics

    def calibrate(self, depth_image, rgb_image):
        """
        
        """
        pass


class CalibrationImageCapturer:

    def __init__(self):
        self.color_img_path = 'images/color_clean_bkg.npy'
        self.depth_img_path = 'images/depth_clean_bkg.npy'

        spatial_filter = rs.spatial_filter()
        spatial_filter.set_option(rs.option.filter_magnitude, 2)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

        hole_filling_filter = rs.hole_filling_filter()
        hole_filling_filter.set_option(rs.option.holes_fill, 1)  # farest_from_around

        self.postprocessing_filters = [
            spatial_filter, hole_filling_filter
        ]

    def capture_and_save(self):
        realsense = RealSenseCameraWrapper(enable_depth=True, enable_color=True)
        color_stream = realsense.get_color_image_source()
        depth_stream = realsense.get_depth_image_source()

        color_img = color_stream.get_new_image()
        depth_img = depth_stream.get_new_image(self.postprocessing_filters)

        np.save(self.depth_img_path, depth_img)
        np.save(self.color_img_path, color_img)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.load(self.depth_img_path), np.load(self.color_img_path)


def find_corners_in_color(color_img):
    gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 3, 3)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=50, param2=20, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(color_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv.circle(color_img, (i[0], i[1]), 2, (0, 0, 255), 3)
    return color_img


def find_corners_in_depth_harris(depth_img):
    gray = cv.convertScaleAbs(depth_img)
    # gray = cv.GaussianBlur(gray, (7, 7), 3, 3)
    color = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
    grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, 3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, 3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_x = cv.convertScaleAbs(grad_x)
    grad_y = cv.convertScaleAbs(grad_y)
    plt.imshow(grad_x);
    plt.show()
    plt.imshow(grad_y);
    plt.show()

    dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    color[dst > 0.01 * dst.max()] = [0, 0, 255]
    return color


def quantize_depth_values(img, quantization_interval=20):
    quantized_img = img / quantization_interval
    quantized_in_original_unis = quantized_img.astype(int) * quantization_interval
    return quantized_in_original_unis


def construct_grid(img_size, portion=0.5, interval_in_pixels=10):
    height = img_size[0]
    width = img_size[1]

    x_half_portion = width * portion / 2
    x_start = int(width / 2 - x_half_portion)
    x_end = int(width / 2 + x_half_portion)

    y_half_portion = height * portion / 2
    y_start = int(height / 2 - y_half_portion)
    y_end = int(height / 2 + y_half_portion)

    x = np.arange(x_start, x_end, interval_in_pixels)
    y = np.arange(y_start, y_end, interval_in_pixels)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return xx, yy


def center_of_mass(image: np.ndarray):
    """
    Calculates the center of mass of the given image.
    Does not take into account the actual values of the pixels,
    but rather treats the pixels as either background, or something.

    Parameters
    ----------
    image : tf.Tensor of shape [width, height, 1]

    Returns
    -------
    center_of_mass : tf.Tensor of shape [3]
        Represented in UVZ coordinates.
        Returns [0,0,0] for zero-sized image, which can happen after a crop
        of zero-sized bounding box.
    """
    # Create all coordinate pairs
    img_shape = image.shape
    im_width = img_shape[0]
    im_height = img_shape[1]
    coords = create_coord_pairs_np(im_width, im_height, indexing='ij')

    image_mask = (image > 0.).astype(float)
    image_mask_flat = np.reshape(image_mask, [im_width * im_height, 1])
    # The total mass of the depth
    total_mass = np.sum(image)
    total_mass = total_mass.astype(float)
    nonzero_pixels = np.count_nonzero(image_mask)
    # Multiply the coords with volumes and reduce to get UV coords
    volumes_vu = np.sum(image_mask_flat * coords, axis=0)
    volumes_uvz = np.stack([volumes_vu[1], volumes_vu[0], total_mass], axis=0)
    if nonzero_pixels == 0:
        return [0, 0]
    else:
        com_uvz = (volumes_uvz / nonzero_pixels).astype(int)
        return com_uvz


def find_corners_in_depth(depth_img, layer_width_mm=40):
    img = quantize_depth_values(depth_img)
    # grid_xx, grid_yy = construct_grid(depth_img.shape)

    # Detect corners in layers
    img_max = np.max(img)
    first_layer_mm = 140
    # Do not take into account the last layer
    num_layers = int((img_max - first_layer_mm) / layer_width_mm)
    last_contour_area = 0
    for layer_idx in range(num_layers):
        layer_min_depth = first_layer_mm + layer_idx * layer_width_mm
        layer_max_depth = first_layer_mm + (layer_idx + 1) * layer_width_mm

        layer_img = img.copy()
        layer_img[(img < layer_min_depth)] = layer_min_depth
        layer_img[(layer_max_depth <= img)] = 0

        gray_layer_img = cv.convertScaleAbs(layer_img)

        drawing = cv.cvtColor(gray_layer_img, cv.COLOR_GRAY2RGB)
        center_of_mass_point = center_of_mass(gray_layer_img)[:2]
        cv.circle(drawing, tuple(center_of_mass_point), 4, (220, 50, 0), -1)

        contours, hierarchy = cv.findContours(gray_layer_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=lambda x: cv.contourArea(x))
        hull = cv.convexHull(biggest_contour, returnPoints=True)
        current_contour_area = cv.contourArea(biggest_contour)

        if current_contour_area > 100 and last_contour_area < current_contour_area:
            cv.drawContours(drawing, [hull], -1, (0, 40, 200), 3)
            plt.imshow(drawing)
            plt.show()
            last_contour_area = current_contour_area
        pass

    # depth_img
    return img


if __name__ == "__main__":
    capturer = CalibrationImageCapturer()
    # capturer.capture_and_save()
    depth_img, color_img = capturer.load()

    # color_img_with_circles = find_corners_in_color(color_img)
    # plt.imshow(color_img_with_circles)
    # plt.show()
    # plt.imshow(depth_img); plt.show()
    # depth_img_with_circles = find_corners_in_depth_harris(depth_img)
    depth_img_another = find_corners_in_depth(depth_img)
    plt.imshow(depth_img_another)
    plt.show()
