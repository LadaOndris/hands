import tensorflow as tf

from src.estimation.jgrp2o.preprocessing import extract_bboxes
from src.system.components.base import KeypointsToRectangle


class KeypointsToRectangleImpl(KeypointsToRectangle):

    def __init__(self):
        pass

    @tf.function
    def convert(self, keypoints_uv):
        bbox = extract_bboxes(keypoints_uv[tf.newaxis, ...], shift_coeff=0.1)[0]
        return bbox