"""
Loads Blazeface model and prints predicted bounding boxers on Handseg dataset.
"""

import tensorflow as tf

from src.datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from src.detection.blazeface.model import build_blaze_face
from src.detection.blazeface.utils import train_utils
from src.detection.blazeface.utils.train_utils import shift_depth
from src.detection.plots import draw_bboxes
from src.estimation.blazepose.data.preprocessing import add_noise
from src.utils import bbox_utils
from src.utils.paths import HANDSEG_DATASET_DIR, MODELS_DIR

hyper_params = train_utils.get_hyper_params()
model = build_blaze_face(hyper_params['detections_per_layer'], channels=1)
print(model.summary(line_length=150))
model.load_weights(MODELS_DIR.joinpath('blazehand.h5'))
# dataset = TvhandDataset(TVHAND_DATASET_DIR, out_img_size=hyper_params['img_size'], batch_size=1)
# dataset = SimpleBoxesDataset(SIMPLE_DATASET_DIR, train_size=0.8,
#                             img_size=[hyper_params['img_size'], hyper_params['img_size']], batch_size=1)
dataset = HandsegDatasetBboxes(
    HANDSEG_DATASET_DIR,
    train_size=0.8,
    img_size=[hyper_params['img_size'], hyper_params['img_size']],
    batch_size=1)
prior_boxes = bbox_utils.generate_prior_boxes(hyper_params['feature_map_shapes'],
                                              hyper_params['aspect_ratios'])

for image, boxes in dataset.test_dataset:
    image_with_noise = add_noise(image[0])
    image_preprocessed = shift_depth(image_with_noise, min_shift=200, max_shift=300)[tf.newaxis, ...]

    pred_deltas, pred_scores = model(image_preprocessed)
    # pred_deltas *= hyperparams['variances']

    actual_deltas, actual_labels = train_utils.calculate_expected_outputs(prior_boxes, boxes, hyper_params)
    pred_bboxes = bbox_utils.get_bboxes_from_deltas(prior_boxes, pred_deltas)
    pred_bboxes = tf.clip_by_value(pred_bboxes, 0, 1)

    pred_scores = tf.cast(pred_scores, tf.float32)

    weighted_bboxes = bbox_utils.weighted_suppression(pred_scores[0], pred_bboxes[0],
                                                      max_total_size=2,
                                                      score_threshold=0.5)

    draw_bboxes(image_preprocessed, weighted_bboxes[tf.newaxis, ...])
    pass
