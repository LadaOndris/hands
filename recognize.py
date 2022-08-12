import argparse

import pyrealsense2 as rs

from src.system.components.coordinate_predictors.MediapipeCoordinatePredictor import MediapipeCoordinatePredictor
from src.system.components.coordinate_predictors.TrackingCoordinatePredictor import TrackingCoordinatePredictor
from src.system.components.detector import BlazehandDetector
from src.system.components.displays.OpencvDisplay import OpencvDisplay
from src.system.components.estimator import BlazeposeEstimator
from src.system.components.gesture_recognizers.MLPGestureRecognizer import MLPGestureRecognizer
from src.system.components.gesture_recognizers.RelativeDistanceGestureRecognizer import \
    RelativeDistanceGestureRecognizer
from src.system.components.image_sources.DefaultVideoCaptureSource import DefaultVideoCaptureSource
from src.system.components.image_sources.RealSenseCameraWrapper import RealSenseCameraWrapper
from src.system.components.keypoints_to_rectangle import KeypointsToRectangleImpl
from src.system.main import System
from src.utils.camera import get_camera
from src.system.components.base import Display
from src.system.components.displays.EmptyDisplay import EmptyDisplay
from src.system.components.displays.StdoutDisplay import StdoutDisplay


def get_depth_filters():
    decimation_filter = rs.decimation_filter()
    decimation_filter.set_option(rs.option.filter_magnitude, 1)

    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 2)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
    spatial_filter.set_option(rs.option.holes_fill, 1)

    postprocessing_filters = [
        spatial_filter
    ]
    return postprocessing_filters


def get_display(display_name: str) -> Display:
    if display_name == 'opencv':
        return OpencvDisplay()
    if display_name == 'stdout':
        return StdoutDisplay()
    if display_name == 'empty':
        return EmptyDisplay()
    raise NameError(f'Invalid display name: {display_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, action='store',
                        help='the name of the directory containg the user-captured gesture database')
    parser.add_argument('--display', type=str, action='store', default='opencv',
                        help='the diplay to be used (available: opencv, stdout, empty, default: opencv)')
    parser.add_argument('--camera', type=str, action='store', default=None,
                        help='the camera model in use for live capture (default: None -> VideoCapture(0) is selected)')
    parser.add_argument('--error-threshold', type=int, action='store', default=150,
                        help='the pose (JRE) threshold (default: 120)')
    parser.add_argument('--orientation-threshold', type=int, action='store', default=90,
                        help='the orientation threshold in angles (maximum: 90, default: 90)')
    args = parser.parse_args()

    if args.camera is None:
        # Uses color camera and MediaPipe tracker
        image_source = DefaultVideoCaptureSource()
        predictor = MediapipeCoordinatePredictor()
    else:
        # Uses depth camera and custom-trained neural nets
        realsense_wrapper = RealSenseCameraWrapper(enable_depth=True, enable_color=False,
                                                   filters=get_depth_filters())
        image_source = realsense_wrapper.get_depth_image_source()
        camera = get_camera(args.camera)
        predictor = TrackingCoordinatePredictor(detector=BlazehandDetector(),
                                                estimator=BlazeposeEstimator(camera, presence_threshold=0.3),
                                                keypoints_to_rectangle=KeypointsToRectangleImpl(shift_coeff=0.1),
                                                camera=camera)
    display = get_display(args.display)
    simple_recognizer = RelativeDistanceGestureRecognizer(args.error_threshold, args.orientation_threshold,
                                                          args.directory)
    regression_recognizer = MLPGestureRecognizer(args.directory)

    system = System(image_source=image_source,
                    predictor=predictor,
                    display=display,
                    gesture_recognizer=regression_recognizer)
    system.start()
