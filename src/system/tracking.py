import argparse
import time

import tensorflow as tf
import pyrealsense2 as rs

from src.system.components.base import Detector, Display, Estimator, GestureRecognizer, ImageSource, \
    KeypointsToRectangle
from src.system.components.detector import BlazehandDetector
from src.system.components.display import OpencvDisplay
from src.system.components.estimator import BlazeposeEstimator
from src.system.components.image_source import RealSenseCameraWrapper
from src.system.components.keypoints_to_rectangle import KeypointsToRectangleImpl
from src.utils.camera import Camera, CameraBighand
from src.system.components.gestures import SimpleGestureRecognizer


class HandTracker:
    """
    HandTracker combines hand detection and pose estimation into a single system.
    Estimation is performed only if there is a detected hand, and
    detection is not enabled unless there is no hand to being tracked.
    """

    def __init__(self, image_source: ImageSource, detector: Detector, estimator: Estimator,
                 keypoints_to_rectangle: KeypointsToRectangle, display: Display, camera: Camera,
                 gesture_recognizer: GestureRecognizer = None):
        self.image_source = image_source
        self.detector = detector
        self.estimator = estimator
        self.keypoints_to_rectangle = keypoints_to_rectangle
        self.display = display
        self.camera = camera
        self.gesture_recognizer = gesture_recognizer

        decimation_filter = rs.decimation_filter()
        decimation_filter.set_option(rs.option.filter_magnitude, 1)

        spatial_filter = rs.spatial_filter()
        spatial_filter.set_option(rs.option.filter_magnitude, 2)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        spatial_filter.set_option(rs.option.holes_fill, 1)

        self.postprocessing_filters = [
            spatial_filter
        ]

    def track(self):
        rectangle = None
        keypoints = None

        while True:
            # Capture image to process next
            start_time = time.time()
            image = self.image_source.get_new_image(self.postprocessing_filters)
            image = tf.convert_to_tensor(image)

            # Detect when there no hand being tracked
            if keypoints is None:
                print("Detecting hand.")
                rectangle = self.detector.detect(image)[0]  # rectangle's max values are [480, 480] (orig image size)
                # self.display.update(image.numpy(), bounding_boxes=[rectangle])

            # Estimate keypoints if there a hand detected
            if not tf.experimental.numpy.allclose(rectangle, 0):
                hand_presence_flag, keypoints, normalized_keypoints, normalized_image, crop_offset_uv = \
                    self.estimator.estimate(image, rectangle)

                # Display the predicted keypoints and prepare for the next frame
                if hand_presence_flag:
                    # self.display.update(image, keypoints)
                    print("Updating window.")
                    rectangle = self.keypoints_to_rectangle.convert(keypoints)
                    # self.display.update(normalized_image.numpy())
                    gesture_label = None
                    if self.gesture_recognizer is not None:
                        keypoints_xyz = self.camera.pixel_to_world(keypoints)
                        gesture_result = self.gesture_recognizer.recognize(keypoints_xyz)
                        if gesture_result.is_gesture_valid:
                            gesture_label = gesture_result.gesture_label
                    self.display.update(image.numpy(), keypoints=keypoints, bounding_boxes=[rectangle],
                                        gesture_label=gesture_label)
                    # yield keypoints.numpy(), normalized_keypoints.numpy(), normalized_image.numpy(), crop_offset_uv.numpy()
                # Reject if the hand is not present
                else:
                    print("Hand was lost.")
                    self.display.update(image.numpy())
                    keypoints = None

            print("Elapsed time [ms]: {:.0f}".format((time.time() - start_time) * 1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    realsense_wrapper = RealSenseCameraWrapper(enable_depth=True, enable_color=False)
    depth_image_source = realsense_wrapper.get_depth_image_source()
    display = OpencvDisplay()
    camera = CameraBighand()

    tracker = HandTracker(image_source=depth_image_source,
                          detector=BlazehandDetector(),
                          estimator=BlazeposeEstimator(camera, presence_threshold=0.3),
                          keypoints_to_rectangle=KeypointsToRectangleImpl(shift_coeff=0.1),
                          display=display, camera=camera,
                          gesture_recognizer=SimpleGestureRecognizer(120, 90, 'demo'))

    # while True:
    #     img = depth_image_source.get_new_image()
    #     display.update(img)
    tracker.track()
