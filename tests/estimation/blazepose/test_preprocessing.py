import tensorflow as tf

from src.estimation.blazepose.data.preprocessing import draw_gaussian_point, generate_heatmaps


class Test(tf.test.TestCase):

    def test_draw_gaussian_point(self):
        image = tf.zeros([256, 256, 1])
        point = [100, 80]  # [x, y]

        heatmap = draw_gaussian_point(image, point, sigma=5)

        self.assertEqual(image.shape, heatmap.shape)
        self.assertEqual(heatmap[point[1], point[0]], tf.constant(1, dtype=tf.float32))

    def test_generate_heatmaps(self):
        points = [[100, 80], [20, 20], [40, 60]]

        heatmaps = generate_heatmaps(points, [256, 256], [64, 64], sigma=3)

        self.assertEqual([64, 64, len(points)], heatmaps.shape)

        self.plot_heatmap(heatmaps[:, :, 0])

    def plot_heatmap(self, heatmap):
        import matplotlib.pyplot as plt
        plt.imshow(heatmap.numpy())
        plt.show()
