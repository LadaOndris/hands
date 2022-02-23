import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from src.postoperations.calibration.compute import extrinsics
from src.utils.live import get_depth_unit


def convert_depth_coords_to_color_coords(depth_pixel_coords, depth_pixel_value):
    print(F"\n\t depth_pixel: {depth_pixel_coords}, value: {depth_pixel_value}")
    # From 2D space to 3D space
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_pixel_coords, depth_pixel_value)

    # Use custom extrinsic parameters approximated using cv::solvePnP
    extrinsics_in_meters = extrinsics()
    color_point = np.matmul(extrinsics_in_meters, np.array(depth_point + [1]))[0:3]
    # Use extrinsic parameters provided by the manufacturer
    # color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)

    # From 3D space to 2D space
    color_pixel = rs.rs2_project_point_to_pixel(color_intrinsics, color_point)
    print(F"\n\t color_pixel: {color_pixel}")
    return color_pixel


def on_mouse(event, x, y, flags, param):
    global depth_pixel_coords
    if event == cv.EVENT_MOUSEMOVE:
        depth_pixel_coords = (x, y)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480)

profile = pipeline.start(config)

depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
depth_intrinsics = depth_stream.get_intrinsics()
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
color_intrinsics = color_stream.get_intrinsics()

depth_to_color_extrin = depth_stream.get_extrinsics_to(color_stream)
color_to_depth_extrin = color_stream.get_extrinsics_to(depth_stream)

print("\n Depth intrinsics: " + str(depth_intrinsics))
print("\n Color intrinsics: " + str(color_intrinsics))
print("\n Depth to color extrinsics: " + str(depth_to_color_extrin))

depth_unit = get_depth_unit(profile)  # SR305 returns 0.00012498 (mm)
millimeter = 0.001
# depth_unit_correction_factor = depth_unit / millimeter
depth_unit_correction_factor = 1 / depth_unit

depth_pixel_coords = [200, 200]  # Random pixel

cv.namedWindow('Depth', cv.WINDOW_NORMAL)
cv.namedWindow('Color', cv.WINDOW_GUI_NORMAL)
cv.setMouseCallback('Depth', on_mouse)

try:
    while True:
        frameset = pipeline.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()

        depth_image = np.array(depth_frame.get_data())
        color_image = np.array(color_frame.get_data())

        # to meters
        # depth_image[depth_image > 1000] = 0
        depth_image_in_meters = depth_image * depth_unit
        depth_image_in_meters = np.where(depth_image == 0, 0, depth_image_in_meters)[..., np.newaxis]

        depth_drawing = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=255 / depth_image.max()),
                                         cv.COLORMAP_JET)
        color_drawing = color_image

        depth_pixel_value = depth_image_in_meters[depth_pixel_coords[1], depth_pixel_coords[0]]

        color_pixel_coords = convert_depth_coords_to_color_coords(depth_pixel_coords, depth_pixel_value)
        color_pixel_coords = (int(color_pixel_coords[0]), int(color_pixel_coords[1]))
        # color_pixel_value = color_image[color_pixel_coords[1], color_pixel_coords[0]]

        cv.circle(depth_drawing, tuple(depth_pixel_coords), 3, [255, 0, 0], 2)
        cv.circle(color_drawing, tuple(color_pixel_coords), 3, [255, 0, 0], 2)

        cv.imshow('Depth', depth_drawing)
        cv.imshow('Color', color_drawing)

        cv.waitKey(1)
finally:
    pipeline.stop()
    cv.destroyAllWindows()
