import tensorflow as tf

from src.datasets.tvhand.dataset import TvhandDataset
from src.utils.paths import TVHAND_DATASET_DIR
from src.utils import bbox_utils


class TestTvhandDataset(tf.test.TestCase):

    def test_build_dataset(self):
        batch_size = 4
        dataset = TvhandDataset(TVHAND_DATASET_DIR, 256, batch_size=batch_size)
        iterator = iter(dataset.train_dataset)
        image, boxes = next(iterator)

        self.assertAllInRange(boxes, 0, 1)
        self.assertAllInRange(image, 0, 1)

        self.assertEqual([batch_size, 256, 256, 3], image.shape)

    def test_prepare_output(self):
        batch_size = 4
        prior_boxes = bbox_utils.generate_prior_boxes([64, 32, 16, 8, 8, 8], [[1.], [1.], [1.], [1.], [1.], [1.]])
        dataset = TvhandDataset(TVHAND_DATASET_DIR, 256, batch_size=batch_size, prepare_output=True,
                                prior_boxes=prior_boxes)
        iterator = iter(dataset.train_dataset)
        image, outputs = next(iterator)

        boxes, labels = outputs

        self.assertEqual([batch_size, 11136, 4], boxes.shape)
        self.assertEqual([batch_size, 11136, 1], labels.shape)
