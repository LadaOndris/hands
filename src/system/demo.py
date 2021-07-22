import typing

import numpy as np
import tensorflow as tf

import src.estimation.configuration as configs
import src.utils.plots as plots
from src.acceptance.base import fit_hyperplane, hand_orientation, joint_relation_errors, vectors_angle
from src.acceptance.gesture_acceptance_result import GestureAcceptanceResult
from src.detection.plots import image_plot
from src.system.database.reader import UsecaseDatabaseReader
from src.system.hand_position_estimator import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.imaging import average_nonzero_depth, create_coord_pairs


class GestureRecognizer:

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
            typing.Generator[GestureAcceptanceResult, None, None]:
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

            joints_uvz = self.estimator.estimate_from_image(image_array)

            # Detection failed, continue to next image
            if joints_uvz is None:
                continue
            joints_xyz = self.camera.pixel_to_world(joints_uvz)
            acceptance_result = self.accept_gesture(joints_xyz)
            if generator_includes_labels:
                acceptance_result.expected_gesture_label = gesture_label.numpy()

            # plot the hand position with gesture label
            image_subregion = self.estimator.get_cropped_image()
            joints_subregion = self.estimator.convert_to_cropped_coords(joints_uvz)

            mean_depth = average_nonzero_depth(image_subregion[tf.newaxis, ...])[0]
            tf.print(F"Depth: {mean_depth / 10:.0f} cm")

            norm, mean = self.fit_plane_through_hand(image_subregion)
            # norm, mean = acceptance_result.orientation, acceptance_result.orientation_joints_mean
            # norm, mean = self._transform_orientation_vectors_to_2d(norm, mean)
            if self.plot_hand_only:
                self.plot(acceptance_result, image_subregion, joints_subregion, norm, mean)
            else:
                norm = tf.squeeze(self.estimator.convert_to_global_coords(norm))
                mean = tf.squeeze(self.estimator.convert_to_global_coords(mean))
                self.plot(acceptance_result, self.estimator.resized_image, joints_uvz, norm, mean)

            image_idx += 1
            yield acceptance_result

    @staticmethod
    def fit_plane_through_hand(image):
        uvz_coords = GestureRecognizer._setup_uvz_coordinates(image)
        uvz_coords_nonzero = GestureRecognizer._select_random_nonzero_pixels(uvz_coords, samples=50)
        norm3d, mean3d = fit_hyperplane(uvz_coords_nonzero.numpy())
        norm3d = GestureRecognizer._correct_vector_orientation(norm3d)
        return norm3d[:2], mean3d[:2]

    @staticmethod
    def _setup_uvz_coordinates(image):
        height, width, _ = tf.shape(image)
        uv_coords = create_coord_pairs(width, height, indexing='xy')  # [width * height, 2]
        depth_values = tf.reshape(image, [width * height, 1])
        uvz_coords = tf.concat([uv_coords, depth_values], axis=-1)
        return uvz_coords

    @staticmethod
    def _select_random_nonzero_pixels(uvz_coords, samples):
        uvz_coords_nonzero = tf.boolean_mask(uvz_coords, uvz_coords[:, 2] != 0)
        uvz_coords_nonzero = tf.random.shuffle(uvz_coords_nonzero)
        uvz_coords_nonzero = uvz_coords_nonzero[:samples]
        return uvz_coords_nonzero

    @staticmethod
    def _correct_vector_orientation(vector):
        # Correct orientation of the norm vector
        camera = np.array([0, 0, 1])
        # If the vector is not in the direction of camera
        if np.dot(vector, camera) > 0:
            # Then make it go the other direction
            return -vector
        return vector

    def accept_gesture(self, keypoints: np.ndarray) -> GestureAcceptanceResult:
        """
        Compares given keypoints to the ones stored in the database
        and decides whether the hand satisfies some of the defined gestures.
        Basically performs gesture recognition from the hand's skeleton.

        Parameters
        ----------
        keypoints ndarray of 21 keypoints, shape (batch_size, joints, coords)
        """
        result = GestureAcceptanceResult()
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

    def plot(self, acceptance_result: GestureAcceptanceResult, image, joints, norm=None, mean=None):
        if self.plot_result:
            # if self.plot_orientation and norm is not None and mean is not None:
            #    norm, mean = self._transform_orientation_vectors_to_2d(norm, mean)

            gesture_label = self._get_gesture_label(acceptance_result)
            if self.plot_feedback:
                # get JREs
                jres = acceptance_result.joints_jre[:, acceptance_result.predicted_gesture_idx]

                plots.plot_skeleton_with_jre_live(
                    self.fig, self.ax, image, joints, jres,
                    label=gesture_label, norm_vec=norm, mean_vec=mean)
            else:
                plots.plot_skeleton_with_label_live(self.fig, self.ax, image, joints, gesture_label,
                                                    norm_vec=norm, mean_vec=mean)

    def _get_gesture_label(self, result: GestureAcceptanceResult):
        label = result.gesture_label
        if result.is_gesture_valid:
            return F"Gesture {label}"
        else:
            return "None"

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
        mean_cropped = tf.squeeze(self.estimator.convert_to_cropped_coords(mean2d))
        norm_cropped = tf.squeeze(self.estimator.convert_to_cropped_coords(norm2d))
        return norm_cropped, mean_cropped

    def _get_gesture_labels(self, gesture_indices):
        gesture_indices = gesture_indices
        gesture_labels = self.database_reader.get_label_by_index(gesture_indices)
        return gesture_labels
