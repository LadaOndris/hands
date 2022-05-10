import argparse
import os
import time

import numpy as np
import tensorflow as tf

from src.system.components.base import Detector, Display, Estimator, ImageSource, KeypointsToRectangle
from src.system.components.display import OpencvDisplay
from src.system.components.image_source import RealSenseCameraWrapper
from src.utils.camera import Camera, get_camera
from src.utils.logs import get_current_timestamp, make_dir
from src.utils.paths import USECASE_DATASET_DIR
from system.components.detector import BlazehandDetector
from system.components.estimator import BlazeposeEstimator
from system.components.keypoints_to_rectangle import KeypointsToRectangleImpl


class UsecaseDatabaseScanner:

    def __init__(self, subdir: str, image_source: ImageSource,
                 detector: Detector, estimator: Estimator,
                 keypoints_to_rectangle: KeypointsToRectangle,
                 display: Display, camera: Camera):
        self.subdir = subdir
        self.image_source = image_source
        self.detector = detector
        self.estimator = estimator
        self.keypoints_to_rectangle = keypoints_to_rectangle
        self.display = display
        self.camera = camera

        self.subdir_path = USECASE_DATASET_DIR.joinpath(subdir)

    def scan_into_subdir(self, gesture_label, num_samples, scan_period=1):
        """
        Scans images in intervals specified by 'scan_period',
        and saves estimated joints into a new file with current timestamp
        in a directory specified by subdir.
        """
        file_path = self._prepare_file(gesture_label)
        try:
            with open(file_path, 'a+') as file:
                self._scan_from_source(file, scan_period, num_samples)
        except RuntimeError:
            # If there is no camera, remove the prepared file.
            os.remove(file_path)

    def _scan_from_source(self, file, scan_period, num_samples):
        count = 0
        while count < num_samples:
            time_start = time.time()

            image = self.image_source.get_new_image()
            image = tf.convert_to_tensor(image)

            rectangle = self.detector.detect(image)[0]
            if not tf.experimental.numpy.allclose(rectangle, 0):
                hand_presence_flag, keypoints_uvz, normalized_keypoints_uvz, normalized_image, crop_offset_uv = \
                    self.estimator.estimate(image, rectangle)
                if hand_presence_flag:
                    rectangle = self.keypoints_to_rectangle.convert(keypoints_uvz)
                    self.display.update(image.numpy(), keypoints=keypoints_uvz, bounding_boxes=[rectangle])

                    keypoints_xyz = self.camera.pixel_to_world(keypoints_uvz)
                    self._save_joints_to_file(file, keypoints_xyz)
                    self._wait_till_period(time_start, scan_period)
                    count += 1

    def _prepare_file(self, gesture_label):
        if '_' in gesture_label:
            raise Exception("Label cannot include '_' because it is used a separator.")
        make_dir(self.subdir_path)
        timestamp = get_current_timestamp()
        timestamped_file = self.subdir_path.joinpath(F"{gesture_label}_{timestamp}.txt")
        return timestamped_file

    def _save_joints_to_file(self, file, joints):
        formatted_joints = self._format_joints(joints)
        file.write(F"{formatted_joints}\n")

    def _format_joints(self, joints):
        flattened_joints = np.reshape(joints, [-1]).astype(str)
        formatted_joints = ' '.join(flattened_joints)
        return formatted_joints

    def _wait_till_period(self, start_time_in_seconds, period_in_seconds):
        end_time = time.time()
        duration = end_time - start_time_in_seconds
        if duration * 1.01 < period_in_seconds:
            sleep_till_period = period_in_seconds - duration
            time.sleep(sleep_till_period)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, action='store',
                        help='the name of the directory that should contain the user-captured gesture database')
    parser.add_argument('label', type=str, action='store',
                        help='the label of the gesture that is to be captured')
    parser.add_argument('count', type=int, action='store',
                        help='the number of samples to scan')

    parser.add_argument('--scan-period', type=float, action='store', default=1.0,
                        help='intervals between each capture in seconds (default: 1)')
    parser.add_argument('--camera', type=str, action='store', default='SR305',
                        help='the camera model in use for live capture (default: SR305)')
    parser.add_argument('--hide-plot', action='store_true', default=False,
                        help='hide plots of the captured poses - not recommended')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plot = not args.hide_plot

    realsense_wrapper = RealSenseCameraWrapper(enable_depth=True, enable_color=False)
    depth_image_source = realsense_wrapper.get_depth_image_source()
    display = OpencvDisplay()
    camera = get_camera(args.camera)

    scanner = UsecaseDatabaseScanner(subdir=args.directory,
                                     image_source=depth_image_source,
                                     detector=BlazehandDetector(),
                                     estimator=BlazeposeEstimator(camera, presence_threshold=0.3),
                                     keypoints_to_rectangle=KeypointsToRectangleImpl(shift_coeff=0.1),
                                     display=display, camera=camera)
    scanner.scan_into_subdir(args.label, args.count, scan_period=args.scan_period)
