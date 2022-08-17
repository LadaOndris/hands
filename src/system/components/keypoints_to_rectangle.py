import tensorflow as tf

from src.estimation.preprocessing import extract_bboxes
from src.system.components.base import KeypointsToRectangle


class KeypointsToRectangleImpl(KeypointsToRectangle):

    def __init__(self, shift_coeff=0.1):
        self.shift_coeff = shift_coeff

    @tf.function
    def convert(self, keypoints_uv):
        bbox = extract_bboxes(keypoints_uv[tf.newaxis, ...], shift_coeff=self.shift_coeff)[0]
        return bbox
