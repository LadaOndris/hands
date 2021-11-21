import tensorflow as tf

from src.detection.blazeface.model import build_blaze_face
from src.detection.blazeface.utils import train_utils
from src.detection.yolov3 import utils
from src.detection.yolov3.architecture.loader import YoloLoader
from src.utils import bbox_utils
from src.utils.config import TEST_YOLO_CONF_THRESHOLD
from src.utils.imaging import resize_bilinear_nearest, RESIZE_MODE_CROP, tf_resize_image
from src.utils.paths import LOGS_DIR, SAVED_MODELS_DIR
from system.components.base import Detector


class BlazehandDetector(Detector):

    def __init__(self):
        self.channels = 1
        self.num_detections = 1
        self.hyper_params = train_utils.get_hyper_params()
        self.model = build_blaze_face(self.hyper_params['detections_per_layer'], channels=self.channels)
        self.model.load_weights(LOGS_DIR.joinpath('20211120-142355/train_ckpts/weights.78.h5'))
        self.prior_boxes = bbox_utils.generate_prior_boxes(
            self.hyper_params['feature_map_shapes'],
            self.hyper_params['aspect_ratios'])

    @property
    def input_shape(self):
        return [self.hyper_params['img_size'], self.hyper_params['img_size'], self.channels]

    @tf.function
    def detect(self, image):
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        images = self.preprocess(image)
        deltas_and_scores = self.model(images)
        bboxes = self.postprocess(deltas_and_scores, image_width, image_height)
        return bboxes[0]

    def preprocess(self, image):
        image = resize_bilinear_nearest(image, shape=self.input_shape[:2])
        # image = tf_resize_image(image,
        #                         shape=self.input_shape[:2],
        #                         resize_mode=RESIZE_MODE_CROP)

        batch_images = image[tf.newaxis, ...]
        return batch_images

    def postprocess(self, model_output, original_img_width, original_img_height):
        pred_deltas, pred_scores = model_output

        pred_bboxes = bbox_utils.get_bboxes_from_deltas(self.prior_boxes, pred_deltas)
        pred_bboxes = tf.clip_by_value(pred_bboxes, 0, 1)

        pred_scores = tf.cast(pred_scores, tf.float32)

        weighted_bboxes = bbox_utils.weighted_suppression(pred_scores[0], pred_bboxes[0],
                                                          max_total_size=self.num_detections,
                                                          score_threshold=0.5)
        bboxes = bbox_utils.denormalize_bboxes(weighted_bboxes,
                                               original_img_height,
                                               original_img_width)
        bboxes = tf.stack([bboxes[..., 1], bboxes[..., 0], bboxes[..., 3], bboxes[..., 2]], axis=-1)
        return bboxes[tf.newaxis, ...]

    def save_model(self, path):
        self.model.save(path)


class YoloDetector(Detector):

    def __init__(self, batch_size, resize_mode, num_detections):
        self.num_detections = num_detections
        self.batch_size = batch_size
        self.model = YoloLoader.load_from_weights(resize_mode, batch_size=1)

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


if __name__ == "__main__":
    detector = BlazehandDetector()
    ret = detector.model(tf.zeros([1, 256, 256, 1]))
    print(ret)
    detector.save_model(SAVED_MODELS_DIR.joinpath('blazeface_handseg.tf'))
