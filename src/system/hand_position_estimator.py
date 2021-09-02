import tensorflow as tf

import src.detection.plots
import src.utils.imaging
import src.utils.plots as plots
from src.detection.detector import BlazefaceDetector, YoloDetector
from src.estimation.architecture.jgrp2o import JGR_J2O
from src.estimation.configuration import Config
from src.estimation.preprocessing import convert_coords_to_local, DatasetPreprocessor
from src.utils.camera import Camera
from src.utils.debugging import timing
from src.utils.imaging import read_image_from_file, tf_resize_image
from src.utils.paths import LOGS_DIR


class HandPositionEstimator:
    """
    Loads a hand detector and hand pose estimator.
    Then, it uses them to estimate the precision position of hands
    either from files, live2 images, or from given images.
    """

    def __init__(self, camera: Camera, config: Config, detector: str = 'yolo',
                 plot_detection=False, plot_estimation=False, plot_skeleton=True):
        self.camera = camera
        self.plot_detection = plot_detection
        self.plot_estimation = plot_estimation
        self.plot_skeleton = plot_skeleton
        self.network = JGR_J2O(n_features=196)
        self.estimator = self.load_estimator()
        self.estimation_preprocessor = DatasetPreprocessor(None, self.network.input_size, self.network.out_size,
                                                           camera=self.camera, config=config)
        self.estimation_fig_location = None
        self.resized_image = None
        self.resize_mode = 'crop'
        if detector == 'blazeface':
            self.detector = BlazefaceDetector(num_detections=1)
        elif detector == 'yolo':
            self.detector = YoloDetector(batch_size=1, resize_mode=self.resize_mode, num_detections=1)
        else:
            raise ValueError(F"Invalid detector: {detector}")

        if self.plot_estimation or self.plot_detection:
            # Prepare the plot for live plotting
            self.fig, self.ax = src.detection.plots.image_plot()

    def load_estimator(self):
        # Load HPE model and weights
        # Trained on the BigHand dataset - ideal for live recognition
        weights_path = LOGS_DIR.joinpath("20210426-125059/train_ckpts/weights.25.h5")
        model = self.network.graph()
        model.load_weights(str(weights_path))
        return model

    def estimate_from_file(self, file_path: str, fig_location=None):
        """
        Estimate the hand's pose from a single depth image stored in a file.

        Parameters
        ----------
        file_path : str
            The path to the image.
        fig_location
            A file path defining the location to which save the plot.

        Returns
        -------
        joints_locations
        """
        self.estimation_fig_location = fig_location
        image = self._read_image(file_path)
        return self.estimate_from_image(image)

    def estimate_from_source(self, source_generator, save_folder=None):
        """
        Estimate the hand's pose from images produced by
        the given source_generator.

        Parameters
        ----------
        source_generator
            A generator producing depth images.
        save_folder
            A folder to save the created plots into.

        Returns
        -------
        Generator[joints_locations]
            Yields the positions of the hand's joints.
        """
        i = 0
        for depth_image in source_generator:
            if tf.rank(depth_image) == 4:
                depth_image = tf.squeeze(depth_image, axis=0)
            if save_folder is not None:
                fig_location = save_folder.joinpath(F"{i}.png")
                self.estimation_fig_location = fig_location
            joints_uvz, image_subregion, joints_subregion, bboxes = \
                self.estimate_from_image(depth_image)
            if joints_uvz is None:
                continue
            yield joints_uvz
            i += 1

    def estimate_from_image(self, image):
        """
        Estimates the hand's pose from the given depth image.

        Parameters
        ----------
        image
            Image pixels should be in milimeters as they are preprocessed
            accordingly for the detector.

        Returns
        -------
        joints_locations
        """
        image = tf.convert_to_tensor(image)
        tf.debugging.assert_rank(image, 3)

        self.resized_image = self._resize_image(image)
        batch_images = tf.expand_dims(self.resized_image, axis=0)
        boxes = self.detector.detect(batch_images)

        if self._detection_failed(boxes) == tf.constant(True):
            return None, None, None, None
        joints_uvz, image_subregions, joints_subregion, bboxes = self._estimate(batch_images, boxes)
        return joints_uvz, image_subregions[0].to_tensor(), joints_subregion, bboxes

    def detect_from_source(self, source_generator, num_detections: int = 1, fig_location_pattern=None):
        """
        Detects hands in depth images provided by the
        source_generator. Returns only num_detections.

        Parameters
        ----------
        source_generator
            The source of images.
        num_detections : int
            The number of bounding boxes to produce.
        fig_location_pattern
            File path pattern for saving the plot.

        Returns
        -------
        bounding_boxes
            Returns bounding boxes represented by two [u, v] coordinates.
        """
        iter_index = 0
        for depth_image in source_generator:
            if tf.rank(depth_image) == 4:
                depth_image = tf.squeeze(depth_image, axis=0)
            fig_location = self._string_format_or_none(fig_location_pattern, iter_index)
            self.resized_image = self._resize_image(depth_image)
            batch_images = tf.expand_dims(self.resized_image, axis=0)
            boxes = self.detector.detect(batch_images)
            self._plot_detection(batch_images, boxes, fig_location)
            yield boxes
            iter_index += 1
            print(iter_index)

    def get_cropped_image(self):
        """
        Returns only the cropped subimage used for hand pose estimation.
        """
        return self.estimation_preprocessor.cropped_imgs[0].to_tensor()

    def convert_to_cropped_coords(self, joints_uvz):
        """
        Converts the given image coordinates into
        coordinates defined by the cropped image that is used
        for hand pose estimation.

        Parameters
        ----------
        joints_uvz
            Pixel coordinates of the joints' locations.

        Returns
        -------
        converted_joints
        """
        joints_subregion = self.estimation_preprocessor.convert_coords_to_local(joints_uvz)
        return joints_subregion

    def convert_to_global_coords(self, local_uvz_coords):
        return self.estimation_preprocessor.convert_coords_to_global(local_uvz_coords)

    def _string_format_or_none(self, fig_location_pattern, index):
        if fig_location_pattern is None:
            return None
        return str(fig_location_pattern).format(index)

    def _plot_detection(self, images, boxes, fig_location=None):
        if self.plot_detection:
            if fig_location is None:
                src.detection.plots.plot_predictions_live(self.fig, self.ax, images[0], boxes[0])
            else:
                src.detection.plots.plot_predictions(images[0], boxes[0], fig_location)

    @timing
    @tf.function
    def _estimate(self, images, boxes):
        boxes = tf.cast(boxes, dtype=tf.int32)
        normalized_imgs, cropped_imgs, bboxes, bcubes, resize_coeffs = \
            self.estimation_preprocessor.preprocess(images, boxes[:, 0, :])
        # Call predict on the estimator
        y_pred = self.estimator(normalized_imgs)
        uvz_pred = self.estimation_preprocessor.postprocess(y_pred, bboxes, bcubes, resize_coeffs)
        joints_subregion = convert_coords_to_local(uvz_pred, bboxes)
        return uvz_pred, cropped_imgs, joints_subregion, bboxes

    def _plot_estimation(self, uvz_pred):
        # Plot the inferenced pose
        if self.plot_estimation:
            joints2d = self.estimation_preprocessor.convert_coords_to_local(uvz_pred)
            image = self.estimation_preprocessor.cropped_imgs[0].to_tensor()

            if self.estimation_fig_location is None:
                plots.plot_image_with_skeleton_live(self.fig, self.ax, image, joints2d[0])
            else:
                plots.plot_image_with_skeleton(image, joints2d[0], fig_location=self.estimation_fig_location,
                                               figsize=(4, 4))

    def _read_image(self, file_path: str):
        """
        Reads image from file.

        Returns
        -------
        Depth image
        """
        # load image
        depth_image = read_image_from_file(file_path, dtype=tf.uint16, shape=self.camera.image_size)
        return depth_image

    @tf.function
    def _resize_image(self, image):
        """
        Resizes to size matching the detector's input shape.
        The unit of image pixels are set as milimeters.
        """
        image = tf_resize_image(image,
                                shape=self.detector.input_shape[:2],
                                resize_mode=self.resize_mode)
        return image

    def _detection_failed(self, boxes):
        return tf.experimental.numpy.allclose(boxes, 0)
