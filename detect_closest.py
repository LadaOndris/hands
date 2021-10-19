import argparse

import numpy as np
import tensorflow as tf
import src.estimation.jgrp2o.configuration as configs
from src.datasets.generators import get_source_generator
from src.detection.closest.model import ClosestObjectDetector
from src.estimation.jgrp2o.preprocessing_com import ComPreprocessor
from src.utils.camera import Camera
from src.detection.plots import plot_predictions_live, image_plot

parser = argparse.ArgumentParser()
parser.add_argument('source', type=str, action='store',
                    help='the source of images (allowed options: live, dataset)')
parser.add_argument('--camera', type=str, action='store', default='SR305',
                    help='the camera model in use for live capture (default: SR305)')
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot the result of detection')
args = parser.parse_args()

camera = Camera(args.camera)
config = configs.PredictCustomDataset()
image_source = get_source_generator(args.source)
detector = ClosestObjectDetector()
preprocessor = ComPreprocessor(camera)

fig, ax = image_plot()
i = 0
for images in image_source:
    if isinstance(images, tf.Tensor):
        images = images.numpy()
    if len(images.shape) == 3:
        images = images[np.newaxis, ...]
    images = images.astype(np.float32)

    box = detector.detect(images[0])
    if box is None:
        continue
    boxes = box[np.newaxis, ...].astype(np.int32)

    bcubes = preprocessor.refine_bcube_using_com(images, boxes, refine_iters=2).numpy()
    cropped_imgs = preprocessor.crop_bcube(images, bcubes).numpy()
    bboxes = np.concatenate([bcubes[..., 0:2], bcubes[..., 3:5]], axis=-1)

    plot_predictions_live(fig, ax, images[0], boxes)

    i += 1
    print(i, 'Bounding boxes:', bboxes)

