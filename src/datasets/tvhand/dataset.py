import os

import tensorflow as tf
from src.utils.imaging import resize_image_and_bboxes

from src.utils.paths import TVHAND_DATASET_DIR


class TvhandDataset:

    def __init__(self, dataset_path, out_img_size, batch_size=16, train_size=0.8,
                 shuffle=True, augment=False, prepare_output_fn=None, prepare_output_shape=None):
        self.dataset_path = str(dataset_path)
        self.out_img_size = out_img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.batch_index = 0
        self.prepare_output_fn = prepare_output_fn
        self.prepare_output_shape = prepare_output_shape

        annotations = self._load_annotations()
        self.train_annotations, self.test_annotations = self._split_annotations(annotations, train_size)

        self.train_annotations_count = len(self.train_annotations)
        self.test_annotations_count = len(self.test_annotations)
        self.num_train_batches = int(self.train_annotations_count // self.batch_size)
        self.num_test_batches = int(self.test_annotations_count // self.batch_size)

        self.train_dataset = self._build_dataset(self.train_annotations)
        self.test_dataset = self._build_dataset(self.test_annotations)

    def _load_annotations(self):
        annotations_path = os.path.join(self.dataset_path, 'new_annotations.txt')
        with open(annotations_path, 'r') as file:
            annotations = file.readlines()
        return annotations

    def _split_annotations(self, annotations, train_size):
        boundary_index = int(len(annotations) * train_size)
        return annotations[:boundary_index], annotations[boundary_index:]

    def _build_dataset(self, annotations):
        dataset = tf.data.Dataset.from_tensor_slices(annotations)
        if self.shuffle:
            dataset = dataset.shuffle(len(annotations), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._load_image)
        if self.augment:
            dataset = dataset.map(self._augment)
        image_shape = tf.TensorShape([self.out_img_size, self.out_img_size, 3])
        if self.prepare_output_fn:
            dataset = dataset.map(self.prepare_output_fn)
            output_shape = self.prepare_output_shape
        else:
            output_shape = tf.TensorShape([None, 4])
        shapes = (image_shape, output_shape)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=shapes)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    def _load_image(self, annotation_line):
        annotation = tf.strings.strip(annotation_line)  # remove white spaces causing FileNotFound
        annotation_parts = tf.strings.split(annotation, sep=',')
        image_file_name = annotation_parts[0]
        image_file_path = tf.strings.join([self.dataset_path, "/images/", image_file_name])

        file = tf.io.read_file(image_file_path)
        image = tf.io.decode_jpeg(file, channels=3)

        bboxes = tf.reshape(annotation_parts[1:], shape=[-1, 4])
        bboxes = tf.strings.to_number(bboxes, out_type=tf.float32)

        image, bboxes = resize_image_and_bboxes(image, bboxes, [self.out_img_size, self.out_img_size])

        # Normalize values to range [0, 1]
        image /= 255.
        bboxes /= self.out_img_size

        # bboxes seem not to be in correct order. Swap x and y axes.
        bboxes = tf.stack([bboxes[..., 1], bboxes[..., 0], bboxes[..., 3], bboxes[..., 2]], axis=-1)
        return image, bboxes

    def _augment(self, input):
        return input


def merge_annotations():
    """
    Reads annotations.txt and merges lines with the same image name
    because images contain arbitrary number of hands.
    Writes the merged annotations into new_annotations.txt.
    """
    import pandas as pd
    annotations_path = os.path.join(TVHAND_DATASET_DIR, 'annotations.txt')
    new_annotations_path = os.path.join(TVHAND_DATASET_DIR, 'new_annotations.txt')
    df = pd.read_csv(annotations_path, header=None, delimiter=',')
    df_boxes = df.iloc[:, :5]
    merged_lines = df_boxes.groupby(0, as_index=False).agg(lambda x: x.tolist()).to_numpy()

    with open(new_annotations_path, 'w') as file:
        for line in merged_lines:
            file.write(line[0])
            for box_id in range(len(line[1])):
                file.write(F",{line[1][box_id]},{line[2][box_id]},{line[3][box_id]},{line[4][box_id]}")
            file.write('\n')


if __name__ == '__main__':
    ds = TvhandDataset(TVHAND_DATASET_DIR, 360, batch_size=1)
    it = iter(ds.train_dataset)
    img, boxes = next(it)
    import matplotlib.pyplot as plt

    plt.imshow(img[0])
    plt.show()
    pass
