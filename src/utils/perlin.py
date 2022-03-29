import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

@tf.function
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
    delta = tf.cast(delta, tf.float32)
    d = (shape[0] // res[0], shape[1] // res[1])
    res_float = tf.cast(res, tf.float32)

    x_range_float64 = tf.linspace(0, res[0], tf.cast(res_float[0] / delta[0], tf.int32) + 1)[:-1]
    y_range_float64 = tf.linspace(0, res[0], tf.cast(res_float[1] / delta[1], tf.int32) + 1)[:-1]
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


def perlin_mask(img_size, resolution, threshold):
    noise = generate_perlin_noise_2d(img_size, (resolution, resolution))
    mask = noise > threshold
    img = tf.where(mask, 1, 0)
    return img

@tf.function
def random_choice(choices):
    random_number = tf.random.uniform(shape=[1], maxval=1)[0]
    index_float = tf.cast(tf.size(choices), dtype=tf.float32) * random_number
    choice_index_float = tf.math.floor(index_float)
    choice_index_int = tf.cast(choice_index_float, tf.int32)
    return choices[choice_index_int]


def perlin_img_noise(img_size=(256, 256)):
    big_mask_resolution = random_choice(tf.constant([1, 4, 8]))
    big_mask = perlin_mask(img_size, big_mask_resolution, threshold=0)

    small_mask_resolution = random_choice(tf.constant([4, 4, 8, 8, 16, 32, 64, 128]))
    threshold = random_choice(tf.constant([-0.5, -0.4, -0.3, -0.2, -0.1]))
    small_mask = perlin_mask(img_size, small_mask_resolution, threshold=threshold)

    mask = 1 - (1 - small_mask) * big_mask
    return mask


if __name__ == "__main__":
    # noise = generate_perlin_noise_2d((256, 256), (1, 1))
    # noise = noise.numpy()
    noise = perlin_img_noise().numpy()
    plt.imshow(noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.show()
