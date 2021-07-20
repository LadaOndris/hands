from unittest import TestCase

import tensorflow as tf
from src.utils.imaging import average_nonzero_depth


class TestImaging(tf.test.TestCase):

    def test_average_nonzero_depth_invalid_param(self):
        invalid_image_shape = tf.ones(shape=[10, 10, 1])

        with self.assertRaises(TypeError):
            average_nonzero_depth(images=invalid_image_shape)

    def test_average_nonzero_depth_correct_average(self):
        images = tf.ones(shape=[5, 10, 10, 1])
        expected_output = tf.ones(shape=[5])

        actual_output = average_nonzero_depth(images)

        self.assertAllEqual(actual_output, expected_output)

    def test_average_nonzero_depth_correct_output_shape(self):
        expected_shape = tf.ones(shape=[55])
        images = tf.ones(shape=[55, 10, 10, 1])

        actual_output = average_nonzero_depth(images)

        self.assertEqual(tf.shape(actual_output), tf.shape(expected_shape))


    def test_average_nonzero_depth_with_zeros(self):
        image = [1, 1, 0, 0]
        image = tf.convert_to_tensor(image)
        images = tf.reshape(image, [1, 1, -1, 1]) # add dimensions
        images = tf.tile(images, [10, 4, 1, 1]) # create multiples

        actual_output = average_nonzero_depth(images)

        expected_shape = tf.ones(shape=[10])
        self.assertAllEqual(actual_output, expected_shape)