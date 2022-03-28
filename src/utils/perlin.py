import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(shape, res, interpolant=interpolant):
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    x_range_float64 = tf.linspace(0, res[0], int(res[0] / delta[0]) + 1)[:-1]
    y_range_float64 = tf.linspace(0, res[0], int(res[1] / delta[1]) + 1)[:-1]
    x_range = tf.cast(x_range_float64, dtype=tf.float32)
    y_range = tf.cast(y_range_float64, dtype=tf.float32)
    grid_x, grid_y = tf.meshgrid(x_range, y_range)
    grid_x = tf.math.floormod(grid_x, 1)
    grid_y = tf.math.floormod(grid_y, 1)

    # Gradients
    angles_tf = 2 * np.pi * tf.random.uniform(shape=[res[0] + 1, res[1] + 1], minval=0, maxval=1)
    gradients_tf_float64 = tf.stack((tf.math.cos(angles_tf), tf.math.sin(angles_tf)), axis=-1)
    gradients_tf = tf.cast(gradients_tf_float64, dtype=tf.float32)

    gradients_tf = tf.repeat(gradients_tf, d[0], axis=0)
    gradients_tf = tf.repeat(gradients_tf, d[1], axis=1)
    g00_tf = gradients_tf[:-d[0], :-d[1]]
    g10_tf = gradients_tf[d[0]:, :-d[1]]
    g01_tf = gradients_tf[:-d[0], d[1]:]
    g11_tf = gradients_tf[d[0]:, d[1]:]

    # Ramps
    n00_tf = tf.reduce_sum(tf.stack((grid_y, grid_x), axis=-1) * g00_tf, axis=2)
    n10_tf = tf.reduce_sum(tf.stack((grid_y - 1, grid_x), axis=-1) * g10_tf, axis=2)
    n01_tf = tf.reduce_sum(tf.stack((grid_y, grid_x - 1), axis=-1) * g01_tf, axis=2)
    n11_tf = tf.reduce_sum(tf.stack((grid_y - 1, grid_x - 1), axis=-1) * g11_tf, axis=2)

    # Interpolation
    t_x = interpolant(grid_x)
    t_y = interpolant(grid_y)
    n0_tf = n00_tf * (1 - t_y) + t_y * n10_tf
    n1_tf = n01_tf * (1 - t_y) + t_y * n11_tf
    res_tf = np.sqrt(2) * ((1 - t_x) * n0_tf + t_x * n1_tf)
    return res_tf


if __name__ == "__main__":
    noise = generate_perlin_noise_2d((256, 256), (1, 1))
    noise = noise.numpy()
    plt.imshow(noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.show()
