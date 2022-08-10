import os
import time

import numpy as np

from src.system.components import coordinate_predictors
from src.system.components.base import Display, ImageSource, KeypointsToRectangle
from src.utils.logs import get_current_timestamp, make_dir
from src.utils.paths import USECASE_DATASET_DIR


class UsecaseDatabaseScanner:

    def __init__(self, subdir: str, image_source: ImageSource,
                 predictor: coordinate_predictors,
                 keypoints_to_rectangle: KeypointsToRectangle,
                 display: Display):
        self.subdir = subdir
        self.image_source = image_source
        self.predictor = predictor
        self.keypoints_to_rectangle = keypoints_to_rectangle
        self.display = display

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
            prediction = self.predictor.predict(image)

            if prediction is not None:
                self.display.update(image, keypoints=prediction.image_coordinates)

                self._save_joints_to_file(file, prediction.world_coordinates)
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
