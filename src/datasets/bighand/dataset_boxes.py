import tensorflow as tf

from src.datasets.bighand.dataset_base import BighandDatasetBase
from src.utils.imaging import crop_with_pad, resize_bilinear_nearest


class BighandDatasetBoxes(BighandDatasetBase):

    def __init__(self, dataset_path, output_image_shape, test_subject=None, batch_size=16, shuffle=True):
        super().__init__(dataset_path, 'box_annotation', test_subject, batch_size)
        self.output_image_shape = tf.convert_to_tensor(output_image_shape)
        self.shuffle = shuffle

        self.train_dataset = self._build_dataset(self.train_annotation_files, self.train_annotations)
        self.test_dataset = self._build_dataset(self.test_annotation_files, self.test_annotations)

    def _build_dataset(self, annotation_files, annotations_count):
        """ Cannot perform shuffle with less than 1 element"""
        annotations_count = tf.maximum(annotations_count, 1)
        annotations_count = tf.cast(annotations_count, dtype=tf.int64)

        """ Convert to Tensor and shuffle the files """
        annotation_files = tf.constant(annotation_files, dtype=tf.string)
        if self.shuffle:
            annotation_files = tf.random.shuffle(annotation_files)

        dataset = tf.data.TextLineDataset(annotation_files)
        """ Reshuffle the dataset each iteration """
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=annotations_count, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._prepare_sample)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _prepare_sample(self, annotation):
        annotation_parts = tf.strings.split(annotation, sep=' ')
        image_file_path = annotation_parts[0]

        depth_image = tf.io.read_file(image_file_path)
        depth_image = tf.io.decode_image(depth_image, channels=1, dtype=tf.uint16)
        depth_image = tf.cast(depth_image, dtype=tf.float32)
        # Replace values above 1500 mm with background
        depth_image = tf.where(depth_image < 1500, depth_image, 0)

        bboxes = tf.reshape(annotation_parts[1:], shape=[-1, 4])
        bboxes = tf.strings.to_number(bboxes, out_type=tf.float32)
        normalized_boxes = bboxes / 480.

        # Square images - also update boxes to match them
        cropped_image = crop_with_pad(depth_image, [0, 0, 480, 480])
        crop_delta = (640 - 480) / 640.
        cropped_boxes = normalized_boxes - tf.constant([crop_delta, 0., crop_delta, 0.])
        cropped_boxes = tf.clip_by_value(cropped_boxes, 0., 1.)

        resized_image = resize_bilinear_nearest(cropped_image, self.output_image_shape)
        return resized_image, cropped_boxes
