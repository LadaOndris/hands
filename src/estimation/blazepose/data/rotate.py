import numpy as np
import tensorflow as tf


def rotation_angle_from_21_keypoints(keypoints):
    tf.assert_equal(tf.shape(keypoints)[0], 21)

    keypoints = keypoints[:, :2]  # Forget the Z axis
    # 0 - palm
    # 1 2 3 4
    # 5 6 7 8
    # 9 10 11 12 - middle finger
    # 13 14 15 16
    # 17 18 19 20

    hand_direction = keypoints[0] - keypoints[9]
    base_direction = tf.constant([0, 1], hand_direction.dtype)

    angle = vectors_angle(hand_direction, base_direction)

    if hand_direction[1] < 0:
        return 2 * np.math.pi - angle
    return angle


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    # Compute the norm along the last axis (the 3D coords)
    return vector / tf.norm(vector, axis=-1)[..., tf.newaxis]


def vectors_angle(v1, v2):
    """
    Returns the angle between two vectors.
    """
    v1 = tf.convert_to_tensor(v1)
    v2 = tf.convert_to_tensor(v2)

    v1 = unit_vector(v1)
    v2 = unit_vector(v2)

    if tf.rank(v2) > 1:
        product = tf.matmul(v1, v2, transpose_b=True)
    else:
        product = tf.reduce_sum(tf.multiply(v1, v2))
    return tf.math.acos(tf.clip_by_value(product, -1.0, 1.0))


def rotate_tensor(tensor, angle, center=tf.constant([0, 0])):
    center = tf.cast(center, dtype=tensor.dtype)

    cos = tf.math.cos
    sin = tf.math.sin
    rotation_matrix = tf.stack([cos(angle), -sin(angle),
                                sin(angle), cos(angle)])
    rotation_matrix = tf.reshape(rotation_matrix, [2, 2])

    tensor -= center
    rotated_tensor = tf.matmul(tensor[tf.newaxis, :], rotation_matrix)[0]
    rotated_tensor += center
    return rotated_tensor
