import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from src.utils.live import get_depth_unit

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

    frameset = pipeline.wait_for_frames()
    depth_frame = frameset.get_depth_frame()
    color_frame = frameset.get_color_frame()

    depth_image = np.array(depth_frame.get_data())
    color_image = np.array(color_frame.get_data())

    depth_unit = get_depth_unit(profile)
    millimeter = 0.001
    # depth_unit_correction_factor = depth_unit / millimeter
    depth_unit_correction_factor = 1 / depth_unit

    # to meters
    depth_image[depth_image > 400] = 0
    depth_image_in_meters = 1 / (depth_image * depth_unit)
    depth_image_in_meters = np.where(depth_image == 0, 0, depth_image_in_meters)[..., np.newaxis]

    # hist = cv.calcHist([depth_image], [0], None, [1000], [250, 400])
    # plt.plot(hist)
    # plt.yscale('log')
    # plt.show()
    laplacian = cv.Laplacian(depth_image, cv.CV_64F)
    plt.imshow(laplacian, cmap='gray')
    plt.show()
    #  drawing = cv.cvtColor(depth_image, cv.COLOR_GRAY2BGR)
    drawing = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.4), cv.COLORMAP_BONE)
    cv.namedWindow('Depth', cv.WINDOW_NORMAL)
    cv.imshow('Depth', drawing)
    cv.waitKey(0)
finally:
    pipeline.stop()
    cv.destroyAllWindows()