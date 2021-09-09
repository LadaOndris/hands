import tensorflow as tf

from src.estimation.blazepose.models.ModelCreator import ModelCreator


class BlazePoseArchitecture(tf.test.TestCase):

    def test_heatmap_model_output_shape(self):
        n_points = 21
        model = ModelCreator.create_model('SIGMOID_HEATMAP_LINEAR_REGRESS_HEATMAP', n_points=n_points)

        input_tensor = tf.random.normal(shape=[1, 256, 256, 1])
        expected_output_shape = tf.TensorShape([1, 64, 64, n_points * 1])

        output_tensor = model(input_tensor)

        self.assertEqual(expected_output_shape, output_tensor.shape)

    def test_regress_model_output_shape(self):
        n_points = 21
        model = ModelCreator.create_model('SIGMOID_HEATMAP_LINEAR_REGRESS_REGRESSION', n_points=n_points)

        input_tensor = tf.random.normal(shape=[1, 256, 256, 1])
        expected_output_shape = tf.TensorShape([1, n_points, 5])

        output_tensor = model(input_tensor)

        self.assertEqual(expected_output_shape, output_tensor.shape)

    def test_two_head_model_output_shape(self):
        n_points = 21
        model = ModelCreator.create_model('SIGMOID_HEATMAP_LINEAR_REGRESS_TWO_HEAD', n_points=n_points)

        input_tensor = tf.random.normal(shape=[1, 256, 256, 1])
        expected_heatmap_output_shape = tf.TensorShape([1, 64, 64, n_points * 1])
        expected_regression_output_shape = tf.TensorShape([1, n_points, 5])

        output_tensor = model(input_tensor)
        joints, heatmap = output_tensor

        self.assertEqual(expected_regression_output_shape, joints.shape)
        self.assertEqual(expected_heatmap_output_shape, heatmap.shape)
