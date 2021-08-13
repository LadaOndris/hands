import tensorflow as tf
import numpy as np

from datasets.dataset_base import DatasetBase
from src.utils.imaging import resize_image_and_bboxes


class SimpleBoxesDataset(DatasetBase):
    """
    The SimpleBoxesDataset is a simple dataset for experimentation
    with an object detector. It is much easier to train than a true
    dataset with real images. The dataset contains images with a solid
    background color and two filled circles of different sizes.
    """

    def __init__(self, dataset_path, train_size, img_size, batch_size=16,
                 prepare_output_fn=None, prepare_output_shape=None):
        super().__init__(dataset_path)
        self.type = type
        self.train_size = train_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.batch_index = 0
        self.prepare_output_fn = prepare_output_fn
        self.prepare_output_shape = prepare_output_shape

        self.train_annotations, self.test_annotations = self.load_annotations('bboxes.txt', self.train_size)
        self.num_train_samples = len(self.train_annotations)
        self.num_test_samples = len(self.test_annotations)
        self.num_train_batches = int(np.math.ceil(self.num_train_samples / self.batch_size))
        self.num_test_batches = int(np.math.ceil(self.num_test_samples / self.batch_size))

        self.train_dataset = self._build_iterator(self.train_annotations)
        self.test_dataset = self._build_iterator(self.test_annotations)

    def _build_iterator(self, annotations):
        dataset = tf.data.Dataset.from_tensor_slices(annotations)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(len(annotations), reshuffle_each_iteration=True)
        dataset = dataset.map(self._prepare_sample)
        image_shape = tf.TensorShape([self.img_size[0], self.img_size[1], 1])
        if self.prepare_output_fn:
            dataset = dataset.map(self.prepare_output_fn)
            output_shape = self.prepare_output_shape
        else:
            output_shape = tf.TensorShape([None, 4])
        shapes = (image_shape, output_shape)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=shapes)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    # annotation line consists of depth_image file name and bounding boxes coordinates
    @tf.function()
    def _prepare_sample(self, annotation):
        annotation_parts = tf.strings.split(annotation, sep=' ')
        image_file_name = annotation_parts[0]
        image_file_path = tf.strings.join([self.dataset_path, "/images/", image_file_name])

        depth_image_file_content = tf.io.read_file(image_file_path)
        # loads depth images and converts values to fit in dtype.uint8
        depth_image = tf.io.decode_png(depth_image_file_content, channels=1)

        bboxes = tf.reshape(annotation_parts[1:], shape=[-1, 4])
        bboxes = tf.strings.to_number(bboxes)

        depth_image, bboxes = resize_image_and_bboxes(depth_image, bboxes, self.img_size)

        # Normalize values to range [0, 1]
        depth_image /= 255
        bboxes /= [self.img_size[0], self.img_size[1], self.img_size[0], self.img_size[1]]

        # Y axis first
        bboxes = tf.stack([bboxes[..., 1], bboxes[..., 0], bboxes[..., 3], bboxes[..., 2]], axis=-1)

        return depth_image, bboxes
