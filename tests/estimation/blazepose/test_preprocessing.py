import math

import tensorflow as tf

from estimation.blazepose.data.rotate import rotation_angle_from_21_keypoints
from src.datasets.bighand.dataset import BIGHAND_DATASET_DIR, BighandDataset
from src.estimation.blazepose.data.preprocessing import draw_gaussian_point, generate_heatmaps, preprocess
from src.utils.camera import Camera
from src.utils.plots import plot_image_with_skeleton


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

    def test_preprocessing_joints_are_rotated(self):
        cam = Camera('bighand')
        ds = BighandDataset(BIGHAND_DATASET_DIR, batch_size=1, shuffle=False)

        test_images = [(1, 10), (1, 100), (2, 10), (2, 100), (3, 10), (3, 100)]
        for file_index, line_index in test_images:
            image_raw, joints_raw = self._load_sample_image(ds, file_index, line_index)
            plot_image_with_skeleton(image_raw, cam.world_to_pixel(joints_raw))
            image, joints, heatmaps = preprocess(image_raw, joints_raw, cam, heatmap_sigma=4, cube_size=180,
                                                 image_target_size=256)
            if joints is None:
                print("Skipping because of out of bounds joints.")
                continue
            plot_image_with_skeleton(image, joints * 256)
            rotation_angle = rotation_angle_from_21_keypoints(joints)
            self.assertAllClose(rotation_angle, 0)
            pass

    def _load_sample_image(self, dataset: BighandDataset, file_index, line_index):
        filepath = dataset.train_annotation_files[file_index]
        with open(filepath, 'r') as file:
            lines = file.readlines()
        line = lines[line_index]
        image, joints = dataset._prepare_sample(line)
        return image, joints
