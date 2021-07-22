from unittest import TestCase

import numpy as np
import tensorflow as tf

from src.system.demo import GestureRecognizer


class TestDemoGestureRecognizer(TestCase):

    def test_fit_plane_through_hand_solid_wall(self):
        image = tf.ones(shape=[300, 300, 1])

        norm, mean = GestureRecognizer.fit_plane_through_hand(image)

        self.assertTrue(np.all(norm == mean))
        self.assertTrue(np.all(mean < [300, 300]))
        self.assertTrue(np.all(mean > [0, 0]))

    def test_fit_plane_through_hand_inclined_wall_along_x_axis(self):
        line = tf.range(1, 301, 1)
        image = tf.tile(line[tf.newaxis, :, tf.newaxis], [300, 1, 1])
        image = tf.cast(image, dtype=tf.float32)

        norm, mean = GestureRecognizer.fit_plane_through_hand(image)

        # The plane is inclined along x axis, therefore, the norm vector
        # keeps y axis unchanged. The norm vector is parallel to x axis.
        self.assertEqual(norm[1], mean[1])

    def test_fit_plane_through_hand_inclined_wall_along_y_axis(self):
        line = tf.range(1, 301, 1)
        image = tf.tile(line[:, tf.newaxis, tf.newaxis], [1, 300, 1])
        image = tf.cast(image, dtype=tf.float32)

        norm, mean = GestureRecognizer.fit_plane_through_hand(image)

        # The plane is inclined along y axis, therefore, the norm vector
        # keeps x axis unchanged. The norm vector is parallel to y axis.
        self.assertEqual(norm[0], mean[0])
