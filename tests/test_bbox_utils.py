from unittest import TestCase
from src.utils import bbox_utils
import tensorflow as tf

class Test(TestCase):

    def test_generate_prior_boxes(self):
        prior_boxes = bbox_utils.generate_prior_boxes([8, 8, 8, 16, 32, 64], [[1.], [1.], [1.], [1.], [1.], [1.]])
        pass