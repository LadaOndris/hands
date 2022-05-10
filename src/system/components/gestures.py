import numpy as np
import tensorflow as tf

from src.acceptance.base import hand_orientation, joint_relation_errors, vectors_angle
from src.acceptance.gesture_acceptance_result import GestureAcceptanceResult
from src.system.database.reader import UsecaseDatabaseReader
from src.system.components.base import GestureRecognizer


class SimpleGestureRecognizer(GestureRecognizer):

    def __init__(self, error_thresh: int, orientation_thresh: int, database_subdir: str):
        self.jre_thresh = error_thresh
        self.orientation_thresh = orientation_thresh

        self.database_reader = UsecaseDatabaseReader()
        self.database_reader.load_from_subdir(database_subdir)
        self.gesture_database = self.database_reader.hand_poses

    def recognize(self, keypoints_xyz: np.ndarray) -> GestureAcceptanceResult:
        """
        Compares given keypoints to the ones stored in the database
        and decides whether the hand satisfies some of the defined gestures.
        Basically performs gesture recognition from the hand's skeleton.

        Parameters
        ----------
        keypoints ndarray of 21 keypoints, shape (batch_size, joints, coords)
        """
        result = GestureAcceptanceResult()
        result.joints_jre = joint_relation_errors(keypoints_xyz, self.gesture_database)
        aggregated_errors = np.sum(result.joints_jre, axis=-1)
        result.predicted_gesture_idx = np.argmin(aggregated_errors, axis=-1)
        result.predicted_gesture = self.gesture_database[result.predicted_gesture_idx, ...]
        result.gesture_jre = tf.squeeze(aggregated_errors[..., result.predicted_gesture_idx])

        result.orientation, result.orientation_joints_mean = hand_orientation(keypoints_xyz)
        result.expected_orientation, _ = hand_orientation(result.predicted_gesture)
        angle_difference = np.rad2deg(vectors_angle(result.expected_orientation, result.orientation))
        result.angle_difference = self._fit_angle_for_both_hands(angle_difference)
        result.gesture_label = self._get_gesture_labels(result.predicted_gesture_idx)[0]
        result.is_gesture_valid = result.gesture_jre <= self.jre_thresh and \
                                  result.angle_difference <= self.orientation_thresh
        return result

    def _fit_angle_for_both_hands(self, angle):
        """
        Do not allow angle above 90 because it is unknown
        which hand is in the image.
        """
        if angle > 90:
            return 180 - angle
        else:
            return angle

    def _get_gesture_labels(self, gesture_indices):
        gesture_labels = self.database_reader.get_label_by_index(gesture_indices)
        return gesture_labels
