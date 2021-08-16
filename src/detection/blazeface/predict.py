import tensorflow as tf

from src.datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from src.datasets.simple_boxes.dataset_bboxes import SimpleBoxesDataset
from src.detection.blazeface.model import build_blaze_face
from src.detection.blazeface.utils import train_utils
from src.detection.plots import draw_bboxes
from src.utils import bbox_utils
from src.utils.paths import HANDSEG_DATASET_DIR, LOGS_DIR, SIMPLE_DATASET_DIR

hyper_params = train_utils.get_hyper_params()
model = build_blaze_face(hyper_params['detections_per_layer'], channels=1)
print(model.summary(line_length=150))
model.load_weights(LOGS_DIR.joinpath('20210816-123035/train_ckpts/weights.88.h5'))
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
    pred_deltas, pred_scores = model(image)
    # pred_deltas *= hyperparams['variances']

    actual_deltas, actual_labels = train_utils.calculate_expected_outputs(prior_boxes, boxes, hyper_params)
    pred_bboxes = bbox_utils.get_bboxes_from_deltas(prior_boxes, pred_deltas)
    pred_bboxes = tf.clip_by_value(pred_bboxes, 0, 1)

    pred_scores = tf.cast(pred_scores, tf.float32)

    weighted_bboxes = bbox_utils.weighted_suppression(pred_scores[0], pred_bboxes[0],
                                                      max_total_size=2,
                                                      score_threshold=0.5)
    # nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections \
    #     = tf.image.combined_non_max_suppression(
    #     pred_bboxes[:, :, tf.newaxis, :], pred_scores,
    #     max_total_size=2, max_output_size_per_class=2)
    # denormalized_bboxes = bbox_utils.denormalize_bboxes(nmsed_boxes,
    #                                                     hyper_params['img_size'],
    #                                                     hyper_params['img_size'])
    # # plot_predictions(image[0], denormalized_bboxes[0], None)
    # # denormalized_bboxes = denormalized_bboxes[tf.newaxis, ...]
    draw_bboxes(image, weighted_bboxes[tf.newaxis, ...])
    pass
