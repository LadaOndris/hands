import tensorflow as tf

from src.detection.yolov3depth.architecture.yolo_depth_layer import YoloDepthLayer


class TestYoloDepthLayer(tf.test.TestCase):

    def test_cells_mean_depths(self):
        img_width = 416
        layer = self.setup_depth_layer(img_width=img_width)

        cells_in_row = 13
        channels = 1
        stride = 32  # img_width / cells_in_row = 416 / 13 = 32
        batch = 4
        images = tf.zeros(shape=[batch, img_width, img_width, channels])
        expected_depth = tf.zeros(shape=[batch, cells_in_row, cells_in_row])

        mean_depth = layer.cells_mean_depths(images, stride)

        self.assertAllEqual(mean_depth, expected_depth)

    def setup_depth_layer(self, img_width=416, anchor_size=3):
        anchors = [[anchor_size, anchor_size]]
        n_classes = 0
        input_layer_shape = [img_width, img_width]
        layer = YoloDepthLayer(anchors, n_classes, input_layer_shape)
        return layer


    def test_scale_anchors(self):
        img_width = 416
        cells_in_row = 13
        batch = 4
        channels = 1
        anchor_size = 10
        anchors = 1
        layer = self.setup_depth_layer(img_width=img_width, anchor_size=anchor_size)
        stride = img_width / cells_in_row
        mean_depth = 200
        images = tf.fill([batch, img_width, img_width, channels], value=mean_depth)

        expected_anchor_size = 1. / mean_depth * anchor_size
        expected_anchors = tf.fill([batch, cells_in_row, cells_in_row, anchors, 2], value=expected_anchor_size)

        scaled_anchors = layer.scale_anchors(images, stride)

        self.assertAllCloseAccordingToType(scaled_anchors, expected_anchors)
