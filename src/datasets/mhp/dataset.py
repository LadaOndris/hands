import os

import tensorflow as tf

from src.utils import plots
from src.utils.camera import Camera
from src.utils.paths import BIGHAND_DATASET_DIR


class MhpDataset:

    def __init__(self, dataset_path, out_img_size, batch_size=16, shuffle=True, prepare_output=False):
        self.dataset_path = dataset_path
        self.out_img_size = out_img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.prepare_output = prepare_output

        self.train_annotation_files, self.test_annotation_files = self._load_annotations(annotation_folder)
        self.train_annotations = self._count_annotations(self.train_annotation_files)
        self.test_annotations = self._count_annotations(self.test_annotation_files)
        self.num_train_batches = int(self.train_annotations // self.batch_size)
        self.num_test_batches = int(self.test_annotations // self.batch_size)

        self.train_dataset = self._build_dataset(self.train_annotation_files, self.train_annotations)
        self.test_dataset = self._build_dataset(self.test_annotation_files, self.test_annotations)

    def _load_annotations(self, annotation_folder):
        pass

    def _prepare_output(self):
        pass

