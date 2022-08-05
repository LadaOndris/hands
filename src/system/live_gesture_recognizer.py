import argparse
import time

import pyrealsense2 as rs

from src.gestures.regression import ClassifierGestureRecognizer
from src.system.components import CoordinatePredictor
from src.system.components.base import Display, GestureRecognizer, ImageSource
from src.system.components.CoordinatePredictor import MediapipeCoordinatePredictor, TrackingCoordinatePredictor
from src.system.components.detector import BlazehandDetector
from src.system.components.display import OpencvDisplay
from src.system.components.estimator import BlazeposeEstimator
from src.system.components.gestures import SimpleGestureRecognizer
from src.system.components.image_source import DefaultVideoCaptureSource, RealSenseCameraWrapper
from src.system.components.keypoints_to_rectangle import KeypointsToRectangleImpl
from src.utils.camera import CameraBighand, get_camera


class LiveGestureRecognizer:
    """
    HandTracker combines hand detection and pose estimation into a single system.
    Estimation is performed only if there is a detected hand, and
    detection is not enabled unless there is no hand to being tracked.
    """

    def __init__(self, image_source: ImageSource, predictor: CoordinatePredictor,
                 display: Display, gesture_recognizer: GestureRecognizer = None):
        self.image_source = image_source
        self.predictor = predictor
        self.display = display
        self.gesture_recognizer = gesture_recognizer

    def start(self):
        while True:
            # Capture image to process next
            start_time = time.time()
            image = self.image_source.get_new_image()

            prediction = self.predictor.predict(image)
            if prediction is None:
                self.display.update(image)
            else:
                gesture_label = None
                if self.gesture_recognizer is not None:
                    gesture_result = self.gesture_recognizer.recognize(prediction.world_coordinates)
                    if gesture_result.is_gesture_valid:
                        gesture_label = gesture_result.gesture_label
                self.display.update(image, keypoints=prediction.image_coordinates,
                                    gesture_label=gesture_label)
            print("Elapsed time [ms]: {:.0f}".format((time.time() - start_time) * 1000))


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, action='store',
                        help='the name of the directory that should contain the user-captured gesture database')
    parser.add_argument('--camera', type=str, action='store', default=None,
                        help='the camera model in use for live capture (default: None -> VideoCapture(0) is selected)')
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

    display = OpencvDisplay()
    simple_recognizer = SimpleGestureRecognizer(150, 90, args.directory)
    regression_recognizer = ClassifierGestureRecognizer(args.directory)

    live_recognizer = LiveGestureRecognizer(image_source=image_source,
                                            predictor=predictor,
                                            display=display,
                                            gesture_recognizer=regression_recognizer)
    live_recognizer.start()
