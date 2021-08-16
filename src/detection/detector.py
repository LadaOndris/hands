from abc import ABC, abstractmethod

import tensorflow as tf

from src.detection.blazeface.model import build_blaze_face
from src.detection.blazeface.utils import train_utils
from src.detection.yolov3 import utils
from src.detection.yolov3.architecture.loader import YoloLoader
from src.utils import bbox_utils
from src.utils.config import TEST_YOLO_CONF_THRESHOLD
from src.utils.debugging import timing
from src.utils.paths import LOGS_DIR


class Detector(ABC):

    def __init__(self, num_detections):
        self.num_detections = num_detections

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @abstractmethod
    def detect(self, images):
        pass

    @abstractmethod
    def preprocess(self, images):
        pass

    @abstractmethod
    def postprocess(self, model_output):
        pass


class BlazefaceDetector(Detector):

    def __init__(self, num_detections):
        super().__init__(num_detections)
        self.channels = 1
        self.hyper_params = train_utils.get_hyper_params()
        self.model = build_blaze_face(self.hyper_params['detections_per_layer'], channels=self.channels)
        self.model.load_weights(LOGS_DIR.joinpath('20210816-123035/train_ckpts/weights.88.h5'))
        self.prior_boxes = bbox_utils.generate_prior_boxes(
            self.hyper_params['feature_map_shapes'],
            self.hyper_params['aspect_ratios'])

    @property
    def input_shape(self):
        return [self.hyper_params['img_size'], self.hyper_params['img_size'], self.channels]

    @timing
    @tf.function
    def detect(self, images):
        deltas_and_scores = self.model(images)
        bboxes = self.postprocess(deltas_and_scores)
        return bboxes

    def preprocess(self, images):
        pass

    @timing
    def postprocess(self, model_output):
        pred_deltas, pred_scores = model_output

        pred_bboxes = bbox_utils.get_bboxes_from_deltas(self.prior_boxes, pred_deltas)
        pred_bboxes = tf.clip_by_value(pred_bboxes, 0, 1)

        pred_scores = tf.cast(pred_scores, tf.float32)

        weighted_bboxes = bbox_utils.weighted_suppression(pred_scores[0], pred_bboxes[0],
                                                          max_total_size=self.num_detections,
                                                          score_threshold=0.5)
        bboxes = bbox_utils.denormalize_bboxes(weighted_bboxes,
                                               self.hyper_params['img_size'],
                                               self.hyper_params['img_size'])
        bboxes = tf.stack([bboxes[..., 1], bboxes[..., 0], bboxes[..., 3], bboxes[..., 2]], axis=-1)
        return bboxes[tf.newaxis, ...]


class YoloDetector(Detector):

    def __init__(self, batch_size, resize_mode, num_detections):
        super().__init__(num_detections)
        self.batch_size = batch_size
        self.model = YoloLoader.load_from_weights(resize_mode, batch_size=1)

    @timing
    @tf.function
    def detect(self, images, num_detections=1):
        """
        Parameters
        ----------
        images
        num_detections
            The number of predicted boxes.
        fig_location
            Path including a file name for saving the figure.

        Returns
        -------
        boxes : shape [batch_size, 4]
            Returns all zeros if non-max suppression did not find any valid boxes.
        """
        detection_batch_images = self.preprocess(images)
        # Call predict on the detector
        yolo_outputs = self.model.tf_model(detection_batch_images)

        boxes = self.postprocess(yolo_outputs)
        return boxes

    def preprocess(self, images):
        """
        Multiplies pixel values by 8 to match the units expected by the detector.
        Converts image dtype to tf.uint8.

        Parameters
        ----------
        images
            Image pixel values are expected to be in milimeters.
        """
        dtype = images.dtype
        # The detector expects unit 0.125 mm, and not 1 mm per unit.
        images = tf.cast(images, dtype=tf.float32)
        images *= 8.0
        images = tf.cast(images, dtype=dtype)
        images = tf.image.convert_image_dtype(images, dtype=tf.uint8)
        return images

    @timing
    def postprocess(self, model_output):
        boxes, scores, nums = utils.boxes_from_yolo_outputs(model_output,
                                                            self.model.batch_size,
                                                            self.model.input_shape,
                                                            TEST_YOLO_CONF_THRESHOLD,
                                                            iou_thresh=.7,
                                                            max_boxes=self.num_detections)
        return boxes

    @property
    def input_shape(self):
        return self.model.input_shape
