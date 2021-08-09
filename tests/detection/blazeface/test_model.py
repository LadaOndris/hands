import tensorflow as tf

from src.detection.blazeface.model import build_blaze_face


class Test(tf.test.TestCase):

    def test_build_blaze_face(self):
        image_batch = tf.random.normal([1, 256, 256, 1])
        detections_per_layer = tf.constant([6, 2, 2, 2])
        model = build_blaze_face(detections_per_layer, channels=1)
        boxes, confs = model(image_batch)

        # 6 * 8 * 8 +
        # 2 * 16 * 16 +
        # 2 * 32 * 32 +
        # 2 * 64 * 64
        # = 11136
        self.assertEqual([1, 11136, 4], boxes.shape)
        self.assertEqual([1, 11136, 1], confs.shape)
