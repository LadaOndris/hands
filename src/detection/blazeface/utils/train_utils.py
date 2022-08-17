import tensorflow as tf

from src.estimation.blazepose.data.preprocessing import add_noise
from src.utils import bbox_utils


def get_hyper_params(**kwargs):
    """Generating hyper params in a dynamic way.
    inputs:
        **kwargs = any value could be updated in the hyper_params
    outputs:
        hyper_params = dictionary
    """
    hyper_params = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "learning_decay_rate": 0.97,
        "img_size": 256,
        "feature_map_shapes": [8, 8, 8, 16, 32, 64],
        "aspect_ratios": [[1.], [1.], [1.], [1.], [1.], [1.]],
        "detections_per_layer": [6, 2, 2, 2],
        "iou_threshold": 0.5,
        "neg_pos_ratio": 3,
        "loc_loss_alpha": 1,
        "variances": [0.1, 0.1, 0.2, 0.2],
    }
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    return hyper_params


def prepare_expected_output_fn(prior_boxes, hyper_params):
    @tf.function
    def _prepare_output(image, boxes):
        image_with_noise = add_noise(image)
        image_preprocessed = shift_depth(image_with_noise)

        # Add batch dimension
        boxes = tf.expand_dims(boxes, axis=0)

        deltas, labels = calculate_expected_outputs(prior_boxes, boxes, hyper_params)

        # We are preparing a single sample -> remove the batch axis.
        deltas = tf.squeeze(deltas, axis=0)
        labels = tf.squeeze(labels, axis=0)

        return image_preprocessed, (deltas, labels)
    return _prepare_output


def shift_depth(image, min_shift=-150, max_shift=150):
    """

    image
    min_shift Value in mm to add to the image
    max_shift Value in mm to add to the image

    Returns
    -------
    """
    shape = tf.ones([tf.rank(image)], dtype=tf.int32)
    random_shift = tf.random.uniform(shape, minval=min_shift, maxval=max_shift)
    # Shift all values in the image
    shifted_image = image + random_shift
    # Crop values below 0 and leave 0 where they were
    shifted_cleaned_image = tf.where((shifted_image < 0) | (shifted_image == random_shift),
                                     0, shifted_image)
    return shifted_cleaned_image


def calculate_expected_outputs(prior_boxes, gt_boxes, hyper_params):
    """Calculate ssd actual output values.
    Batch operations supported.
    inputs:
        prior_boxes = (total_bboxes, [center_x, center_y, width, height])
            these values in normalized format between [0, 1]
        gt_boxes = (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary
    outputs:
        actual_deltas = (batch_size, total_bboxes, [delta_bbox_y, delta_bbox_x, delta_bbox_h, delta_bbox_w])
        actual_labels = (batch_size, total_bboxes, [1 or 0])
    """
    tf.assert_rank(prior_boxes, 2)
    tf.assert_rank(gt_boxes, 3)

    # In case of no ground truth boxes
    total_bboxes = tf.shape(prior_boxes)[0]
    batch_size = tf.shape(gt_boxes)[0]
    gt_box_size = tf.shape(gt_boxes)[1]
    if tf.equal(gt_box_size, 0):
        deltas = tf.zeros(shape=[batch_size, total_bboxes, 4])
        labels = tf.zeros(shape=[batch_size, total_bboxes, 1])
        return deltas, labels

    iou_threshold = hyper_params["iou_threshold"]
    # variances = hyper_params["variances"]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.generate_iou_map(bbox_utils.convert_xywh_to_bboxes(prior_boxes), gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    # IoU above threshold
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    # If no IoU is above threshold, then select the highest IoU
    if tf.reduce_any(pos_cond) == tf.constant(False):
        max_ious_indices = tf.argmax(merged_iou_map, axis=1)
        pos_cond = tf.one_hot(max_ious_indices, depth=total_bboxes, dtype=tf.bool, on_value=True, off_value=False)

    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    actual_deltas = bbox_utils.get_deltas_from_bboxes(prior_boxes, expanded_gt_boxes)
    # actual_deltas /= variances
    actual_labels = tf.expand_dims(tf.cast(pos_cond, dtype=tf.float32), -1)
    return actual_deltas, actual_labels
