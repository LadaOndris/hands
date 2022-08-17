import tensorflow as tf

from src.estimation.blazepose.models.ModelCreator import ModelCreator


class BlazePoseArchitecture(tf.test.TestCase):

    def setUp(self):
        self.n_keypoint_features = 4
        self.n_points = 21

    def test_heatmap_model_output_shape(self):
        model = ModelCreator.create_model('LIGHT_HEATMAP',
                                          n_points=self.n_points,
                                          n_point_features=self.n_keypoint_features)

        input_tensor = tf.random.normal(shape=[1, 256, 256, 1])
        expected_output_shape = tf.TensorShape([1, 64, 64, self.n_points * self.n_keypoint_features])

        output_tensor = model(input_tensor)

        self.assertEqual(expected_output_shape, output_tensor.shape)

    def test_regress_model_output_shape(self):
        n_points = 21
        model = ModelCreator.create_model('LIGHT_REGRESSION',
                                          n_points=self.n_points,
                                          n_point_features=self.n_keypoint_features)

        input_tensor = tf.random.normal(shape=[1, 256, 256, 1])
        expected_output_shape = tf.TensorShape([1, n_points, 3])

        # Expects only 'joints' tensor = 3D coordinates
        output_tensor = model(input_tensor)

        self.assertEqual(expected_output_shape, output_tensor.shape)

    def test_two_head_model_output_shape(self):
        model = ModelCreator.create_model('LIGHT_TWOHEAD',
                                          n_points=self.n_points,
                                          n_point_features=self.n_keypoint_features)

        input_tensor = tf.random.normal(shape=[1, 256, 256, 1])
        expected_regression_output_shape = tf.TensorShape([1, self.n_points, 3])
        expected_heatmap_output_shape = tf.TensorShape([1, 64, 64, self.n_points * self.n_keypoint_features])
        expected_presence_output_shape = tf.TensorShape([1, self.n_points, 1])

        output_tensor = model(input_tensor)
        joints, heatmap, presence = output_tensor

        self.assertEqual(expected_regression_output_shape, joints.shape)
        self.assertEqual(expected_heatmap_output_shape, heatmap.shape)
        self.assertEqual(expected_presence_output_shape, presence.shape)
