import numpy as np
from PIL import Image
import scipy.io
import os
import glob
from src.utils.paths import NYU_DATASET_DIR
from src.utils.plots import plot_depth_image, plot_image_with_skeleton
import tensorflow as tf


class NyuDataset:

    def __init__(self, dataset_path, subfolder, batch_size=16, shuffle=False,
                 prepare_output_fn=None, prepare_output_fn_shape=None):
        self.dataset_path = dataset_path
        self.subfolder = subfolder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prepare_output_fn = prepare_output_fn
        self.prepare_output_fn_shape = prepare_output_fn_shape

        self.file_names = self._load_image_file_names()
        self.joints_uvz = self._load_joints_annot()
        self.dataset = self._build_dataset()

    def _load_image_file_names(self):
        pattern = os.path.join(self.dataset_path, self.subfolder, '*.png')
        image_file_names = glob.glob(pattern)
        return image_file_names

    def _load_joints_annot(self):
        joint_data_file = os.path.join(self.dataset_path, self.subfolder, 'joint_data.mat')
        joint_data = scipy.io.loadmat(joint_data_file)
        joints_uvz = joint_data['joint_uvd']
        return tf.convert_to_tensor(joints_uvz)

    def _build_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.file_names)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.file_names), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._prepare_sample)

        if self.prepare_output_fn is not None:
            dataset = dataset.map(self.prepare_output_fn)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _prepare_sample(self, image_path):
        file_path_splits = tf.strings.split(image_path, '.')
        file_path_without_extension = file_path_splits[-2]
        file_name_splits = tf.strings.split(file_path_without_extension, '_')

        kinect_index = tf.strings.to_number(file_name_splits[-2], tf.int32)
        image_index = tf.strings.to_number(file_name_splits[-1], tf.int32)

        depth_image = tf.io.read_file(image_path)
        depth_image = tf.io.decode_image(depth_image, channels=3, dtype=tf.uint8)
        depth_image.set_shape([480, 640, 3])
        depth_image = tf.cast(depth_image, tf.int16)

        green = tf.cast(depth_image[..., 1:2], tf.int16)
        blue = tf.cast(depth_image[..., 2:3], tf.int16)
        depth_image = tf.bitwise.left_shift(green, 8) + blue
        depth_image = tf.cast(depth_image, tf.uint16)

        joints_uvz = self.joints_uvz[kinect_index, image_index, :, :]

        return depth_image, joints_uvz

if __name__ == "__main__":
    ds = NyuDataset(NYU_DATASET_DIR, 'train', batch_size=1)
    for image, joints in ds.dataset:
        plot_image_with_skeleton(image[0], joints[0])
        pass
