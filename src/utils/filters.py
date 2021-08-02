import tensorflow as tf
import tensorflow_probability as tfp


def has_single_color(image):
    image_min = tf.reduce_min(image)
    image_max = tf.reduce_max(image)
    return image_min == image_max


def is_valid_image(image):
    if tf.size(image) == 0 or has_single_color(image):
        return False
    else:
        return True


def apply_threshold(image_in, allowance_threshold):
    """
    Apply otsu thresholding to remove background and leave closest object only
    However if there is another object in similar distance as a hand,
    then it fails to remove it. Then it is probably best to
    find the biggest countour.
    """
    if tf.size(image_in) == 0:
        return image_in
    if type(image_in) is tf.RaggedTensor:
        image = image_in.to_tensor()

    image_above_min = flattened_with_min_value_discarded(image)

    if not is_valid_image(image_above_min):
        return tf.RaggedTensor.from_tensor(image, ragged_rank=2)

    threshold_depth = threshold_otsu(image_above_min)
    threshold_frequency, peak_frequency = threshold_and_peak_frequency(image_above_min, threshold_depth)
    if threshold_frequency / peak_frequency < allowance_threshold:
        image = tf.where(image > threshold_depth, 0., image)

    return tf.RaggedTensor.from_tensor(image, ragged_rank=2)


def flattened_with_min_value_discarded(arr):
    image_min = tf.reduce_min(arr)
    indices = tf.where(arr > image_min)
    image_above_min = tf.gather_nd(arr, indices)
    return image_above_min


def threshold_and_peak_frequency(image: tf.Tensor, threshold_depth):
    # Apply thresholding only if the threshold_frequency
    # is significantly low
    unique, indices, counts = tf.unique_with_counts(image)
    peak_freq = unique[tf.argmax(counts)]
    threshold_freq = counts[unique == tf.math.round(threshold_depth)]
    if tf.size(threshold_freq) == 0:
        threshold_freq = 0
    return tf.cast(threshold_freq, tf.float32), tf.cast(peak_freq, tf.float32)


def threshold_otsu(image, nbins=256):
    """
    Return threshold value based on Otsu's method.

    Reimplementation of the `threshold_otsu` function from
    `skimage.filters` package in Tensorflow.
    """
    min_value = tf.reduce_min(image)
    max_value = tf.reduce_max(image)
    tf.debugging.assert_none_equal(min_value, max_value,
                                   message="threshold_otsu is expected to work with images "
                                           "having more than one color. The input image seems "
                                           "to have just one color.")
    bin_edges = histogram_bin_edges(min_value, max_value, nbins)
    hist = tfp.stats.histogram(image, bin_edges)
    # hist = tf.histogram_fixed_width_bins(image, [min_value, max_value], nbins=nbins)
    bin_centers = histogram_bin_centers(min_value, max_value, nbins)
    hist = tf.cast(hist, dtype=tf.float32)

    # class probabilities for all possible thresholds
    weight1 = tf.math.cumsum(hist)
    weight2 = tf.math.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = tf.math.cumsum(hist * bin_centers) / weight1
    mean2 = (tf.math.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = tf.math.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def histogram_bin_edges(min, max, nbins):
    return tf.linspace(min, max, nbins + 1)


def histogram_bin_centers(min, max, nbins):
    bin_edges = tf.linspace(min, max, nbins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    return centers
