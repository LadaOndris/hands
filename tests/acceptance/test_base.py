from unittest import TestCase

import numpy as np

from src.gestures.relative_distance.base import vectors_angle


class TestBase(TestCase):

    def test_vectors_angle_3d_array(self):
        # Setup input vectors
        v1, _ = self._get_vectors_2d()
        v1 = np.tile(v1, (10, 1, 1))  # shape is (10, 4, 3)
        v2 = v1[...]

        # Setup the expected result
        expected_angles = self._get_expected_result_2d()
        expected_angles = np.tile(expected_angles, (10, 1, 1))

        # angle between each combination of vectors for each sample in the batch
        # (10, 4, 3) x (10, 4, 3) => (10, 4, 4)
        angles = vectors_angle(v1, v2)

        self.assertEqual(angles.shape, (10, 4, 4))
        np.testing.assert_array_equal(angles, expected_angles)

    def test_vectors_angle_2d_array(self):
        # Setup input vectors
        v1, v2 = self._get_vectors_2d()

        # Setup the expected result
        expected_angles = self._get_expected_result_2d()

        # angle between each combination of vectors
        # (4, 3) x (4, 3) => (4, 4)
        angles = vectors_angle(v1, v2)

        self.assertEqual(angles.shape, (4, 4))
        np.testing.assert_array_equal(angles, expected_angles)

    def test_vectors_angle_1d_array(self):
        # Setup input vectors
        v1, v2 = self._get_vectors_2d()
        # Get 1D vectors
        idx1 = 0
        idx2 = 3
        v1 = v1[idx1]
        v2 = v2[idx2]

        # Setup the expected result
        expected_angles = self._get_expected_result_2d()
        expected_angle = expected_angles[idx1, idx2]

        # angle between the two vectors
        # (3,) x (3,) => scalar
        angle = vectors_angle(v1, v2)

        np.testing.assert_array_equal(angle, expected_angle)

    def _get_vectors_2d(self):
        vec1 = [0, 0, 1]
        vec2 = [1, 0, 0]
        vec3 = [0, 1, 0]
        vec4 = np.array([1, 1, 1]) * (1. / np.sqrt(3))
        v = np.array([vec1, vec2, vec3, vec4])
        v1 = v[...]  # shape is (4, 3)
        v2 = v[...]  # shape is (4, 3)
        return v1, v2

    def _get_expected_result_2d(self):
        pihalf = np.math.pi / 2
        var = np.arccos(1 / np.sqrt(3))  # 0.9553166181
        expected_angles = np.array([
            [0, pihalf, pihalf, var],
            [pihalf, 0, pihalf, var],
            [pihalf, pihalf, 0, var],
            [var, var, var, 0],
        ])
        return expected_angles

