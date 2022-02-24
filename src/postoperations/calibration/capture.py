"""
A script for capturing points for calibration of two video streams.

The script saves the image pixel coordinates and depth value in files
for later calibration.

The window is closed by Escape key.

"""

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from src.utils.live import get_depth_unit

depth_uvz = []
color_uv = []


def intrinsics_to_matrix(intr):
    return np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])


def on_depth_mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        depth_pixel_value = depth_image_in_mm[y, x]
        print(x, y, depth_pixel_value)
        depth_uvz.append([x, y, depth_pixel_value])


def on_color_mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)
        color_uv.append([x, y])


cv.namedWindow('Depth', cv.WINDOW_NORMAL)
cv.namedWindow('Color', cv.WINDOW_GUI_NORMAL)
cv.setMouseCallback('Depth', on_depth_mouse_click)
cv.setMouseCallback('Color', on_color_mouse_click)

pipeline = rs.pipeline()
try:
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480)

    profile = pipeline.start(config)

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intrinsics = depth_stream.get_intrinsics()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_stream.get_intrinsics()

    np.save("depth_intrinsics.npy", intrinsics_to_matrix(depth_intrinsics))
    np.save("color_intrinsics.npy", intrinsics_to_matrix(color_intrinsics))

    depth_to_color_extrin = depth_stream.get_extrinsics_to(color_stream)
    color_to_depth_extrin = color_stream.get_extrinsics_to(depth_stream)

    print("\n Depth intrinsics: " + str(depth_intrinsics))
    print("\n Color intrinsics: " + str(color_intrinsics))
    print("\n Depth to color extrinsics: " + str(depth_to_color_extrin))
    while True:
        frameset = pipeline.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()

        depth_image = np.array(depth_frame.get_data())
        color_image = np.array(color_frame.get_data())

        depth_unit = get_depth_unit(profile)
        millimeter = 0.001
        depth_unit_correction_factor = depth_unit / millimeter
        depth_image_in_mm = depth_image * depth_unit_correction_factor
        # depth_unit_correction_factor = 1 / depth_unit

        # to meters
        # depth_image[depth_image > 1000] = 0
        # depth_image_in_meters = 1 / (depth_image * depth_unit)
        # depth_image_in_meters = np.where(depth_image == 0, 0, depth_image_in_meters)[..., np.newaxis]
        depth_drawing = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        cv.imshow('Depth', depth_drawing)
        cv.imshow('Color', color_image)

        key = cv.waitKey(1)
        if key % 256 == 27:  # Escape key
            print("========================================")
            print(depth_uvz)
            print(color_uv)
            np.save("depth_uvz.npy", depth_uvz)
            np.save("color_uv.npy", color_uv)
            break
finally:
    pipeline.stop()
    cv.destroyAllWindows()
