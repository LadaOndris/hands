import tensorflow as tf


class BlazeFaceLoss:
    """
    Source:
    https://github.com/FurkanOM/tf-blazeface/blob/dce7aa2b9750b63a0e3611ccc18150e59f220645/ssd_loss.py#L3
    """

    def __init__(self, neg_pos_ratio, loc_loss_alpha):
        self.neg_pos_ratio = tf.constant(neg_pos_ratio, dtype=tf.float32)
        self.loc_loss_alpha = tf.constant(loc_loss_alpha, dtype=tf.float32)

    def box_loss(self, actual_bbox_deltas, pred_bbox_deltas):
        """Calculating SSD localization loss value for only positive samples.
        inputs:
            actual_bbox_deltas = (batch_size, total_bboxes, [delta_bbox_y, delta_bbox_x, delta_bbox_h, delta_bbox_w])
            pred_bbox_deltas = (batch_size, total_bboxes, [delta_bbox_y, delta_bbox_x, delta_bbox_h, delta_bbox_w])
        outputs:
            loc_loss = localization / regression / bounding box loss value
        """

        total_reg_points = tf.shape(actual_bbox_deltas)[-1]
        # Localization / bbox / regression loss calculation for all bboxes
        loc_loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
        loc_loss_for_all = loc_loss_fn(actual_bbox_deltas, pred_bbox_deltas)
        # After tf 2.2.0 version, the huber calculates mean over the last axis
        loc_loss_for_all = tf.cond(tf.greater(tf.rank(loc_loss_for_all), tf.constant(2)),
                                   lambda: tf.reduce_sum(loc_loss_for_all, axis=-1),
                                   lambda: loc_loss_for_all * tf.cast(total_reg_points, dtype=tf.float32))

        actual_bbox_deltas_wh = actual_bbox_deltas[..., 2:4]
        pos_cond = tf.reduce_any(tf.not_equal(actual_bbox_deltas_wh, tf.constant(0.0)), axis=2)
        pos_mask = tf.cast(pos_cond, dtype=tf.float32)
        total_pos_bboxes = tf.reduce_sum(pos_mask, axis=1)

        loc_loss = tf.reduce_sum(pos_mask * loc_loss_for_all, axis=-1)
        total_pos_bboxes = tf.where(tf.equal(total_pos_bboxes, tf.constant(0.0)), tf.constant(1.0), total_pos_bboxes)
        loc_loss = loc_loss / total_pos_bboxes

        # tf.print("Pos_cond.shape", tf.shape(pos_cond))
        # tf.print("Pos:", total_pos_bboxes)
        # tf.print("Actual deltas:", actual_bbox_deltas[0, :1])
        # tf.print("Predic deltas:", pred_bbox_deltas[0, :1])
        return loc_loss * self.loc_loss_alpha

    def conf_loss(self, actual_labels, pred_labels):
        """Calculating SSD confidence loss value by performing hard negative mining as mentioned in the paper.
        inputs:
            actual_labels = (batch_size, total_bboxes, 1)
            pred_labels = (batch_size, total_bboxes, 1)
        outputs:
            conf_loss = confidence / class / label loss value
        """
        # Confidence / Label loss calculation for all labels
        conf_loss_fn = tf.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)
        conf_loss_for_all = conf_loss_fn(actual_labels, pred_labels)
        #
        squeezed_actual_labels = tf.squeeze(actual_labels, -1)
        pos_cond = tf.not_equal(squeezed_actual_labels, tf.constant(0.0))
        pos_mask = tf.cast(pos_cond, dtype=tf.float32)
        total_pos_bboxes = tf.reduce_sum(pos_mask, axis=1)
        # Hard negative mining
        total_neg_bboxes = tf.cast(total_pos_bboxes * self.neg_pos_ratio, tf.int32)
        #
        masked_loss = tf.where(tf.equal(squeezed_actual_labels, tf.constant(0.0)), conf_loss_for_all,
                               tf.zeros_like(conf_loss_for_all, dtype=tf.float32))
        sorted_loss = tf.argsort(masked_loss, direction="DESCENDING")
        sorted_loss = tf.argsort(sorted_loss)
        neg_cond = tf.less(sorted_loss, tf.expand_dims(total_neg_bboxes, axis=1))
        neg_mask = tf.cast(neg_cond, dtype=tf.float32)
        #
        final_mask = pos_mask + neg_mask
        conf_loss = tf.reduce_sum(final_mask * conf_loss_for_all, axis=-1)
        total_pos_bboxes = tf.where(tf.equal(total_pos_bboxes, tf.constant(0.0)), tf.constant(1.0), total_pos_bboxes)
        conf_loss = conf_loss / total_pos_bboxes
        #
        return conf_loss
