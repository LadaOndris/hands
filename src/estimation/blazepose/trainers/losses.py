import math

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

epsilon = 1e-5
smooth = 1


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def confusion(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return prec, recall


def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
    return tp


def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    return tn


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow tensor
    :param y_pred: TensorFlow tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def wing_loss(landmarks, labels, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('wing_loss'):
        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss


def get_huber_loss2(delta=1.0, weights=1.0):
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/huber_loss

    def huber_loss(y_true, y_pred):
        return tf.compat.v1.losses.huber_loss(
            y_true, y_pred, weights=weights, delta=delta
        )

    return huber_loss


def get_wing_loss(w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """

    def wing_loss(y_true, y_pred):
        with tf.name_scope('wing_loss'):
            x = y_pred - y_true
            c = w * (1.0 - math.log(1.0 + w / epsilon))
            absolute_x = tf.abs(x)
            losses = tf.where(
                tf.greater(w, absolute_x),
                w * tf.log(1.0 + absolute_x / epsilon),
                absolute_x - c
            )
            loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
            return loss


class AdjustedCoordsLoss(tf.keras.losses.Loss):

    def __init__(self, coord_loss):
        super().__init__()
        self.coord_loss = coord_loss

    def call(self, y_true, y_pred):
        """Make sure that loss is computed only if the hand is present."""
        presence = y_true[..., 3:4]
        true_keypoints = y_true[..., :3] * presence
        pred_keypoints = y_pred[..., :3] * presence
        coords_loss_val = self.coord_loss(true_keypoints, pred_keypoints)
        # coords_loss_val_mean = tf.reduce_mean(coords_loss_val, axis=-1)
        return coords_loss_val


class HeatmapLossDecorator(tf.keras.losses.Loss):

    def __init__(self, heatmap_loss, offsets_loss, num_joints):
        super().__init__()
        self.heatmap_loss = heatmap_loss
        self.offsets_loss = offsets_loss
        self.num_joints = num_joints

    def call(self, y_true, pred_heatmaps_and_offsets):
        """Make sure that loss is computed only if the hand is present."""
        true_heatmaps_and_offsets = y_true[..., :-1]
        presence = y_true[..., -1:]

        true_heatmaps_and_offsets *= presence
        pred_heatmaps_and_offsets *= presence

        true_heatmap = true_heatmaps_and_offsets[..., :self.num_joints]
        pred_heatmap = pred_heatmaps_and_offsets[..., :self.num_joints]
        heatmaps_loss_val = self.heatmap_loss(true_heatmap, pred_heatmap)

        true_offsets = true_heatmaps_and_offsets[..., self.num_joints:4 * self.num_joints]
        pred_offsets = pred_heatmaps_and_offsets[..., self.num_joints:4 * self.num_joints]
        offsets_loss_val = self.offsets_loss(true_offsets, pred_offsets)

        # Compute coords loss if the joint is present
        # tf.print(tf.shape(coords_loss_val), tf.shape(true_heatmap), tf.shape(presence), tf.shape(pred_heatmap))
        return heatmaps_loss_val + offsets_loss_val
