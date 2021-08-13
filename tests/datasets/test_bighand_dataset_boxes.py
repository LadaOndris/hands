import tensorflow as tf

from src.datasets.bighand.dataset_boxes import BighandDatasetBoxes
from src.utils.paths import BIGHAND_DATASET_DIR


class TestBighandDatasetBoxes(tf.test.TestCase):

    def test_build_dataset(self):
        dataset = BighandDatasetBoxes(BIGHAND_DATASET_DIR, [256, 256])
        iterator = iter(dataset.train_dataset)
        image, boxes = next(iterator)

        self.assertAllGreaterEqual(boxes, 0.)
        self.assertAllLessEqual(boxes, 1.)

        self.assertEqual(image.shape, [256, 256, 1])
