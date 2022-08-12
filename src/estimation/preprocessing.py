import tensorflow as tf


def extract_bboxes(uv_coords, shift_coeff=0.2):
    """
    Parameters
    ----------
    uv_global   tf.Tensor of shape [None, n_joints, 2]

    Returns
    -------
        tf.Tensor of shape [None, 4].
        The four values are [left, top, right, bottom].
    """
    tf.assert_rank(uv_coords, 3)

    # Extract u and v coordinates
    u = uv_coords[..., 0]
    v = uv_coords[..., 1]

    # Find min and max over joints axis.
    u_min, u_max = tf.reduce_min(u, axis=1), tf.reduce_max(u, axis=1)
    v_min, v_max = tf.reduce_min(v, axis=1), tf.reduce_max(v, axis=1)

    # Move min and max to make the bbox slighty larger
    width = u_max - u_min
    height = v_max - v_min
    u_shift = width * shift_coeff
    v_shift = height * shift_coeff
    u_min, u_max = u_min - u_shift, u_max + u_shift
    v_min, v_max = v_min - v_shift, v_max + v_shift

    # The bounding box is represented by four coordinates
    bboxes = tf.stack([u_min, v_min, u_max, v_max], axis=-1)
    bboxes = tf.cast(bboxes, dtype=tf.int32)
    return bboxes


def get_resize_coeffs(bboxes, target_size):
    cropped_imgs_sizes_u = bboxes[..., 2] - bboxes[..., 0]  # right - left
    cropped_imgs_sizes_v = bboxes[..., 3] - bboxes[..., 1]  # bottom - top
    cropped_imgs_sizes = tf.stack([cropped_imgs_sizes_u, cropped_imgs_sizes_v], axis=-1)
    return tf.cast(target_size / cropped_imgs_sizes, dtype=tf.float32)


def resize_coords(coords_uv, resize_coeffs):
    resized_uv = resize_coeffs[..., tf.newaxis, :] * coords_uv
    return resized_uv


def get_uv_offsets(joints, offsets_size):
    n_joints = joints.shape[0]
    print(joints)
    x = offset_coord(joints[..., 0], n_joints, offsets_size)  # shape = [None, out_size, n_joints]
    y = offset_coord(joints[..., 1], n_joints, offsets_size)  # shape = [None, out_size, n_joints]

    u_offsets = x[tf.newaxis, :, :]
    u_offsets = tf.tile(u_offsets, [offsets_size, 1, 1])

    v_offsets = y[:, tf.newaxis, :]
    v_offsets = tf.tile(v_offsets, [1, offsets_size, 1])
    return u_offsets, v_offsets


def offset_coord(joints_single_coord, n_joints, image_size):
    x = tf.linspace(0, 1, num=image_size)  # [0, 1]
    x = tf.cast(x, tf.float32)
    x = x[:, tf.newaxis]  # shape = [out_size, 1]
    x = tf.tile(x, [1, n_joints])  # shape = [out_size, n_joints]
    x -= joints_single_coord  # [0 - u, 1 - u], shape = [ out_size, n_joints]
    x *= -1  # [u, u - 1]
    return x