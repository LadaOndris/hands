import os

import tensorflow as tf

from src.utils.imaging import set_depth_unit, zoom_in


class HandsegDatasetBboxes:

    def __init__(self, dataset_path, train_size, img_size, batch_size=16, shuffle=True, augment=False,
                 prepare_output_fn=None, prepare_output_shape=None):
        if train_size < 0 or train_size > 1:
            raise ValueError("Train_size expected to be in range [0, 1], but got {train_size}.")

        self.dataset_path = str(dataset_path)
        self.train_size = train_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.batch_index = 0
        self.prepare_output_fn = prepare_output_fn
        self.prepare_output_shape = prepare_output_shape

        self.train_annotations, self.test_annotations = self._load_annotations()
        self.num_train_batches = int(len(self.train_annotations) // self.batch_size)
        self.num_test_batches = int(len(self.test_annotations) // self.batch_size)

        self.train_dataset = self._build_iterator(self.train_annotations)
        self.test_dataset = self._build_iterator(self.test_annotations)

    def _load_annotations(self):
        annotations_path = os.path.join(self.dataset_path, 'bounding_boxes.txt')
        with open(annotations_path, 'r') as f:
            annotations = f.readlines()

        boundary_index = int(len(annotations) * self.train_size)
        return annotations[:boundary_index], annotations[boundary_index:]

    def _build_iterator(self, annotations):
        dataset = tf.data.Dataset.from_tensor_slices(annotations)
        if self.shuffle:
            dataset = dataset.shuffle(len(annotations), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._prepare_sample)
        if self.augment:
            dataset = dataset.map(self._augment)
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
        annotation = tf.strings.strip(annotation)  # remove white spaces causing FileNotFound
        annotation_parts = tf.strings.split(annotation, sep=' ')
        image_file_name = annotation_parts[0]
        image_file_path = tf.strings.join([self.dataset_path, "/images/", image_file_name])

        # depth_image = tf.keras.preprocessing.image.load_img(image_file_name, color_mode='grayscale')
        depth_image = tf.io.read_file(image_file_path)
        # loads depth images and converts values to fit in dtype.uint8
        depth_image = tf.io.decode_image(depth_image, channels=1, dtype=tf.uint16)
        # Correct depth unit
        unit_mm = 0.001
        unit_previous = 0.00012498664727900177  # 0.125 mm
        depth_image = set_depth_unit(depth_image, target_depth_unit=unit_mm,
                                     previous_depth_unit=unit_previous)
        depth_image = tf.cast(depth_image, dtype=tf.float32)
        # Replace values above 1500 mm with background
        depth_image = tf.where(depth_image < 1500, depth_image, 0)

        depth_image.set_shape([480, 640, 1])
        bboxes = tf.reshape(annotation_parts[1:], shape=[-1, 4])
        bboxes = tf.strings.to_number(bboxes)

        depth_image, bboxes = self.crop(depth_image, bboxes)
        # Normalize to [0, 1] range
        bboxes /= [self.img_size[0], self.img_size[1], self.img_size[0], self.img_size[1]]
        # Y axis first
        bboxes = tf.stack([bboxes[..., 1], bboxes[..., 0], bboxes[..., 3], bboxes[..., 2]], axis=-1)
        # depth_image, bboxes = self.pad(depth_image, bboxes)
        return depth_image, bboxes

    def crop(self, depth_image, bboxes):
        depth_image = depth_image[tf.newaxis, ...]
        depth_image = tf.image.crop_and_resize(depth_image, [[0, 80 / 640.0, 480 / 480.0, 560 / 640.0]],
                                               [0], self.img_size[:2])
        depth_image = depth_image[0]
        # move bboxes
        bboxes = bboxes - tf.constant([80, 0, 80, 0], dtype=tf.float32)[tf.newaxis, :]
        # crop out of bounds boxes
        bboxes = tf.clip_by_value(bboxes, 0., 479.)
        # remove too narrow boxes because of the crop
        bboxes_mask_indices = tf.where(bboxes[..., 2] - bboxes[..., 0] > 5.)
        bboxes = tf.gather_nd(bboxes, bboxes_mask_indices)
        # normalize bboxes
        bboxes *= self.img_size[0] / 480
        return depth_image, bboxes

    def pad(self, depth_image, bboxes):
        depth_image = tf.image.resize_with_pad(depth_image, 416, 416)
        m = tf.tile(tf.constant([[416 / 640, 416 / 640, 416 / 640, 416 / 640]], dtype=tf.float32),
                    [tf.shape(bboxes)[0], 1])
        a = tf.tile(tf.constant([[0, 52, 0, 52]], dtype=tf.float32), [tf.shape(bboxes)[0], 1])
        bboxes = tf.math.multiply(bboxes, m)
        bboxes = tf.math.add(bboxes, a)
        return depth_image, bboxes

    def _augment(self, depth_image, bboxes):
        new_distance = tf.random.uniform(shape=[1], minval=50, maxval=1500)
        tf.print(tf.shape(bboxes))
        if tf.shape(bboxes)[0] != 0:
            depth_image, bboxes = zoom_in(depth_image, bboxes, new_distance=new_distance[0])
        return depth_image, bboxes
