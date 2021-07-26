from unittest import TestCase
import tensorflow as tf
from src.utils.imaging import average_nonzero_depth, zoom_in


class Test(tf.test.TestCase):

    def setUp(self):
        x1, x2, y1, y2 = 50, 100, 100, 170
        self.bboxes = tf.constant([[x1, y1, x2, y2]])
        self.box_width = x2 - x1
        self.box_height = y2 - y1

        self.distance = 500.
        box_image = tf.fill([self.box_height, self.box_width, 1], self.distance)
        self.image = tf.image.pad_to_bounding_box(box_image, y1, x1, 416, 416)

    def test_zoom_in(self):
        target_distance = 250.
        new_image, new_bboxes = zoom_in(self.image, self.bboxes, self.distance / target_distance)

        self.assertAllEqual(tf.shape(new_image), [416, 416, 1])
        self.assertEqual(target_distance, average_nonzero_depth(new_image[tf.newaxis, ...])[0])

    def test_zoom_out(self):
        target_distance = 1000.
        new_image, new_bboxes = zoom_in(self.image, self.bboxes, self.distance / target_distance)

        self.assertEqual(target_distance, average_nonzero_depth(new_image[tf.newaxis, ...])[0])

    def test_zoom_too_much_no_bbox(self):
        target_distance = tf.constant(5.)
        new_image, new_bboxes = zoom_in(self.image, self.bboxes, self.distance / target_distance)

        # self.assertAlmostEqual(target_distance, average_nonzero_depth(new_image[tf.newaxis, ...])[0])
        self.assertEqual(0, tf.shape(new_bboxes)[0])

    def test_zoom_in_bbox_twice_as_big(self):
        target_distance = 250.
        zoom = self.distance / target_distance
        new_box_width = tf.cast(self.box_width * zoom, tf.int32)
        new_box_height = tf.cast(self.box_height * zoom, tf.int32)

        new_image, new_bboxes = zoom_in(self.image, self.bboxes, zoom)

        bbox = new_bboxes[0]
        self.assertEqual(new_box_width, bbox[2] - bbox[0])
        self.assertEqual(new_box_height, bbox[3] - bbox[1])

    def test_zoom_in_bbox_twice_as_small(self):
        target_distance = 1000.
        zoom = self.distance / target_distance
        new_box_width = tf.cast(self.box_width * zoom, tf.int32)
        new_box_height = tf.cast(self.box_height * zoom, tf.int32)

        new_image, new_bboxes = zoom_in(self.image, self.bboxes, zoom)

        bbox = new_bboxes[0]
        self.assertEqual(new_box_width, bbox[2] - bbox[0])
        self.assertEqual(new_box_height, bbox[3] - bbox[1])