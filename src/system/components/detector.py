import tensorflow as tf

from src.detection.blazeface.model import build_blaze_face
from src.detection.blazeface.utils import train_utils
from src.system.components.base import Detector
from src.utils import bbox_utils
from src.utils.imaging import resize_bilinear_nearest
from src.utils.paths import MODELS_DIR, SAVED_MODELS_DIR


class BlazehandDetector(Detector):

    def __init__(self):
        self.channels = 1
        self.num_detections = 1
        self.hyper_params = train_utils.get_hyper_params()
        self.model = build_blaze_face(self.hyper_params['detections_per_layer'], channels=self.channels)
        self.model.load_weights(MODELS_DIR.joinpath('blazehand.h5'))
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


if __name__ == "__main__":
    detector = BlazehandDetector()
    ret = detector.model(tf.zeros([1, 256, 256, 1]))
    print(ret)
    detector.save_model(SAVED_MODELS_DIR.joinpath('blazeface_handseg.tf'))
