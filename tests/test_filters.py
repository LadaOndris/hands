import tensorflow as tf
from skimage import exposure, filters

from src.utils.filters import histogram_bin_centers, threshold_otsu


class Test(tf.test.TestCase):

    def test_bin_centers(self):
        nbins = 16
        max_val = 255
        array = tf.range(0, max_val + 1, delta=1, dtype=tf.float32)
        centers = histogram_bin_centers(0, max_val, nbins)
        hist, expected_centers = exposure.histogram(array.numpy(), nbins=nbins)
        self.assertAllClose(centers, expected_centers)

    def test_threshold_otsu(self):
        nbins = 16
        max_val = 255
        array = tf.range(0, max_val + 1, delta=1, dtype=tf.float32)
        threshold = threshold_otsu(array, 16)
        expected_threshold = filters.threshold_otsu(array.numpy(), nbins=nbins)
        self.assertAllClose(threshold, expected_threshold)
