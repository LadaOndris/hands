import os

import tensorflow as tf

from src.datasets.bighand.dataset_base import BighandDatasetBase
from src.utils import plots
from src.utils.camera import Camera, CameraBighand
from src.utils.paths import BIGHAND_DATASET_DIR


class BighandDataset(BighandDatasetBase):

    def __init__(self, dataset_path, test_subject=None, batch_size=16, shuffle=True,
                 prepare_output_fn=None):
        super().__init__(dataset_path, 'full_annotation', test_subject, batch_size)
        self.shuffle = shuffle
        self.batch_index = 0
        self.prepare_output_fn = prepare_output_fn
        self.camera = CameraBighand()

        self.train_dataset = self._build_dataset(self.train_annotation_files, self.train_annotations)
        self.test_dataset = self._build_dataset(self.test_annotation_files, self.test_annotations)

    def _build_dataset_one_sample(self, annotation_files):
        file = annotation_files[0]
        tf.print(file)
        with open(file, 'r') as f:
            annotation_line = f.readline()
        line = tf.constant(annotation_line, dtype=tf.string)
        line = tf.reshape(line, shape=[1])
        ds = tf.data.Dataset.from_tensor_slices(line)
        ds = ds.repeat()
        ds = ds.map(self._prepare_sample)
        ds = ds.batch(1)
        return ds

    def _build_dataset(self, annotation_files, annotations_count):
        """ Read specified files """
        # dataset = tf.data.Dataset.from_tensor_slices(annotations)

        """ Read all available annotations """
        # pattern = os.path.join(self.dataset_path, 'full_annotation/*/*.txt')
        # dataset = tf.data.Dataset.list_files(pattern)

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

        if self.prepare_output_fn is not None:
            dataset = dataset.filter(self._joints_are_in_bounds)
            dataset = dataset.map(self.prepare_output_fn)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _joints_are_in_bounds(self, image, kp_xyz):
        image_shape = tf.cast(tf.shape(image), kp_xyz.dtype)
        height, width = image_shape[0], image_shape[1]
        kp_uvz = self.camera.world_to_pixel_2d(kp_xyz)
        u = kp_uvz[..., 0]
        v = kp_uvz[..., 1]
        mask = (u < 0) | (v < 0) | (u > width) | (v > height)
        num_invalid = tf.math.count_nonzero(mask)
        are_in_bounds = tf.cast(num_invalid, tf.float64) < tf.shape(kp_uvz)[0] / 3
        return are_in_bounds

    def _prepare_sample(self, annotation_line):
        """ Each line contains 64 values: file_name, 21 (joints) x 3 (coords) """

        """ If the function processes a single line """
        splits = tf.strings.split(annotation_line, sep='\t', maxsplit=63)  # Split by whitespaces
        filename, labels = tf.split(splits, [1, 63], 0)
        joints = tf.strings.to_number(labels, tf.float32)
        joints = tf.reshape(joints, [21, 3])
        """
        # If the function processes a batch
        splits = splits.to_tensor()
        filename, labels = tf.split(splits, [1, 63], 1)
        """

        """ Compose a full path to the image """
        image_paths = tf.strings.join([tf.constant(str(self.dataset_path)), filename], separator=os.sep)
        """ Squeeze the arrays dimension if necessary"""
        image_paths = tf.squeeze(image_paths)

        """ Read and decode image (tf doesn't support for more than a single image)"""
        depth_image = tf.io.read_file(image_paths)
        depth_image = tf.io.decode_image(depth_image, channels=1, dtype=tf.uint16)
        depth_image.set_shape([480, 640, 1])

        """ Reorder joints """
        reorder_idx = tf.constant([
            0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20
        ], dtype=tf.int32)
        joints = tf.gather(joints, reorder_idx)

        return depth_image, joints


if __name__ == '__main__':
    cam = CameraBighand()
    ds = BighandDataset(BIGHAND_DATASET_DIR, batch_size=10, shuffle=True)
    iterator = iter(ds.train_dataset)
    batch_images, batch_labels = next(iterator)

    for image, joints in zip(batch_images, batch_labels):
        image = tf.squeeze(image)
        joints2d = cam.world_to_pixel(joints)
        plots.plot_image_with_skeleton(image, joints2d)
        pass
