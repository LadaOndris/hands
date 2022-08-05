import typing

import numpy as np
import tensorflow as tf

import estimation.jgrp2o.configuration as configs
import src.utils.plots as plots
from src.acceptance.base import fit_plane_through_hand, hand_orientation, joint_relation_errors, \
    vectors_angle
from src.acceptance.gesture_acceptance_result import GestureRecognitionResult
from src.detection.plots import image_plot
from src.estimation.jgrp2o.preprocessing import convert_coords_to_global
from src.system.database.reader import UsecaseDatabaseReader
from src.system.hand_position_estimator import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.debugging import timing
from src.utils.imaging import average_nonzero_depth


class ModifiedLiveGestureRecognizer:

    def __init__(self, error_thresh: int, orientation_thresh: int, database_subdir: str, camera_name: str,
                 plot_result=True, plot_feedback=False, plot_orientation=True, plot_hand_only=False):
        self.jre_thresh = error_thresh
        self.orientation_thresh = orientation_thresh
        self.plot_result = plot_result
        self.plot_feedback = plot_feedback
        self.plot_orientation = plot_orientation
        self.plot_hand_only = plot_hand_only

        self.camera = Camera(camera_name)
        config = configs.PredictCustomDataset()
        self.estimator = HandPositionEstimator(self.camera, config=config)
        self.database_reader = UsecaseDatabaseReader()
        self.database_reader.load_from_subdir(database_subdir)
        self.gesture_database = self.database_reader.hand_poses

    def start(self, image_generator, generator_includes_labels=False) -> \
            typing.Generator[GestureRecognitionResult, None, None]:
        """
        Starts gesture recognition. It uses images supplied by
        image_generator.

        Parameters
        ----------
        image_generator : generator
            The source of images.
        generator_includes_labels : bool
            Whether the generator also returns labels.

        Returns
        -------
        Generator[GestureAcceptanceResult]
            Yields instances of the GestureAcceptanceResult class.
        """
        image_idx = 0
        self.prepare_plot()

        for image_array in image_generator:
            # If the generator also returns labels, expand the tuple
            if generator_includes_labels:
                image_array, gesture_label = image_array
            if tf.rank(image_array) == 4:
                image_array = image_array[0]

            joints_uvz, image_subregion, joints_subregion, bboxes = \
                self.estimator.estimate_from_image(image_array)

            # Detection failed, continue to next image
            if joints_uvz is None:
                continue
            joints_xyz = self.camera.pixel_to_world(joints_uvz)
            acceptance_result = self.accept_gesture(joints_xyz)
            if generator_includes_labels:
                acceptance_result.expected_gesture_label = gesture_label.numpy()

            mean_depth = average_nonzero_depth(image_subregion[tf.newaxis, ...])[0]
            tf.print(F"Depth: {mean_depth / 10:.0f} cm")

            norm, mean = fit_plane_through_hand(image_subregion)
            # norm, mean = acceptance_result.orientation, acceptance_result.orientation_joints_mean
            # norm, mean = self._transform_orientation_vectors_to_2d(norm, mean)
            if self.plot_hand_only:
                self.plot(acceptance_result, image_subregion, joints_subregion, norm, mean)
            else:
                mean = tf.squeeze(convert_coords_to_global(mean, bboxes))
                self.plot(acceptance_result, self.estimator.resized_image, joints_uvz, norm, mean)

            image_idx += 1
            print(image_idx)
            yield acceptance_result

    @timing
    def accept_gesture(self, keypoints: np.ndarray) -> GestureRecognitionResult:
        """
        Compares given keypoints to the ones stored in the database
        and decides whether the hand satisfies some of the defined gestures.
        Basically performs gesture recognition from the hand's skeleton.

        Parameters
        ----------
        keypoints ndarray of 21 keypoints, shape (batch_size, joints, coords)
        """
        result = GestureRecognitionResult()
        result.joints_jre = joint_relation_errors(keypoints, self.gesture_database)
        aggregated_errors = np.sum(result.joints_jre, axis=-1)
        result.predicted_gesture_idx = np.argmin(aggregated_errors, axis=-1)
        result.predicted_gesture = self.gesture_database[result.predicted_gesture_idx, ...]
        result.gesture_jre = tf.squeeze(aggregated_errors[..., result.predicted_gesture_idx])

        result.orientation, result.orientation_joints_mean = hand_orientation(keypoints)
        result.expected_orientation, _ = hand_orientation(result.predicted_gesture)
        angle_difference = np.rad2deg(vectors_angle(result.expected_orientation, result.orientation))
        result.angle_difference = self._fit_angle_for_both_hands(angle_difference)
        result.gesture_label = self._get_gesture_labels(result.predicted_gesture_idx)[0]
        result.is_gesture_valid = result.gesture_jre <= self.jre_thresh and \
                                  result.angle_difference <= self.orientation_thresh
        return result

    def prepare_plot(self):
        # Prepare figure for live plotting, but only if we are really going to plot.
        if self.plot_result:
            if self.plot_feedback:
                self.fig, self.ax = plots.plot_skeleton_with_jre_subplots()
            else:
                self.fig, self.ax = image_plot()

    def plot(self, acceptance_result: GestureRecognitionResult, image, joints, norm=None, mean=None):
        if self.plot_result:
            # if self.plot_orientation and norm is not None and mean is not None:
            #    norm, mean = self._transform_orientation_vectors_to_2d(norm, mean)

            gesture_label = acceptance_result.get_gesture_label()
            if self.plot_feedback:
                # get JREs
                jres = acceptance_result.joints_jre[:, acceptance_result.predicted_gesture_idx]

                plots.plot_skeleton_with_jre_live(
                    self.fig, self.ax, image, joints, jres,
                    label=gesture_label, norm_vec=norm, mean_vec=mean)
            else:
                plots.plot_skeleton_with_label_live(self.fig, self.ax, image, joints, gesture_label,
                                                    norm_vec=norm, mean_vec=mean)

    def _fit_angle_for_both_hands(self, angle):
        """
        Do not allow angle above 90 because it is unknown
        which hand is in the image.
        """
        if angle > 90:
            return 180 - angle
        else:
            return angle

    def _transform_orientation_vectors_to_2d(self, norm3d, mean3d):
        norm2d, mean2d = self.camera.world_to_pixel(
            np.stack([mean3d + 20 * norm3d, mean3d]))
        norm2d -= mean2d
        mean_cropped = tf.squeeze(self.estimator.convert_to_cropped_coords(mean2d))
        norm_cropped = tf.squeeze(self.estimator.convert_to_cropped_coords(norm2d))
        return norm_cropped, mean_cropped

    def _get_gesture_labels(self, gesture_indices):
        gesture_indices = gesture_indices
        gesture_labels = self.database_reader.get_label_by_index(gesture_indices)
        return gesture_labels
