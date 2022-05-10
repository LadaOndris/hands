from typing import Tuple

import numpy as np
import tensorflow as tf


class Camera:

    def __init__(self,
                 focal_length: Tuple[float, float],
                 principal_point: Tuple[float, float],
                 image_size: Tuple[int, int],
                 extrinsic_matrix: np.ndarray = np.eye(4),
                 depth_unit: float = None):
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.depth_unit = depth_unit
        self.extrinsic_matrix = tf.constant(extrinsic_matrix, dtype=tf.float32)

        self.intrinsic_matrix = tf.constant([[self.focal_length[0], 0, self.principal_point[0], 0],
                                             [0, self.focal_length[1], self.principal_point[1], 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=tf.float32)
        # World_to_pixel projection matrix
        self.projection_matrix = tf.matmul(self.intrinsic_matrix, self.extrinsic_matrix)
        # Pixel_to_world projection matrix
        self.invr_projection_matrix = tf.linalg.inv(self.projection_matrix)

    def set_msra_creative_interactive(self):
        self.focal_length = [241.42, 241.42]
        self.principal_point = [160, 120]
        self.image_size = [320, 240]

    def set_bighand_sr300(self):
        self.focal_length = [475.065948, 475.065857]  # [476.0068, 476.0068]  # [588.235, 587.084]
        self.principal_point = [315.944855, 245.287079]
        self.image_size = [640, 480]
        self.extrinsic_matrix = tf.constant(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 0],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 0],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 0],
             [0, 0, 0, 1]], dtype=tf.float32)
        self.depth_unit = 0.001

    def set_sr305(self):
        # self.focal_length = [588.235, 587.084]
        self.focal_length = [476.0068, 476.0068]
        self.principal_point = [313.6830139, 242.7547302]
        self.image_size = [640, 480]
        self.depth_unit = 0.00012498664727900177  # 0.125 mm

    def set_d415(self):
        self.focal_length = [592.138, 592.138]
        self.principal_point = [313.79, 238.076]
        self.image_size = [640, 480]
        self.depth_unit = 0.001  # 1 mm

    def create_projection_matrices(self):
        # More information on intrinsic and extrinsic parameters
        # can be found at: https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters
        """
        self.extrinsic_matrix = tf.constant(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
             [0, 0, 0, 1]], dtype=tf.float32)

        self.extrinsic_matrix = tf.constant(
            [[1, 0, 0, 25.7],
             [0, 1, 0, 1.22],
             [0, 0, 1, 3.902],
             [0, 0, 0, 1]], dtype=tf.float32)
        """
        raise RuntimeError('Not supported, deprecated')

    def world_to_pixel(self, coords_xyz):
        """
        Projects the given points through pinhole camera on a plane at a distance of focal_length.
        Returns
        -------
        tf.Tensor
            UVZ coordinates
        """
        if tf.rank(coords_xyz) == 1:
            points_xyz = coords_xyz[tf.newaxis, tf.newaxis, ...]
        elif tf.rank(coords_xyz) == 2:
            points_xyz = coords_xyz[tf.newaxis, ...]
        else:
            points_xyz = coords_xyz

        uvz = self.world_to_pixel_3d(points_xyz)

        if tf.rank(coords_xyz) == 1:
            uvz = tf.squeeze(uvz, axis=[0, 1])
        if tf.rank(coords_xyz) == 2:
            uvz = tf.squeeze(uvz, axis=0)
        return uvz

    def world_to_pixel_1d(self, coords_xyz):
        """
        Projects the given points through pinhole camera on a plane at a distance of focal_length.
        Returns
        -------
        tf.Tensor
            UVZ coordinates
        """
        tf.assert_rank(coords_xyz, 1)
        points_xyz = coords_xyz[tf.newaxis, tf.newaxis, ...]
        uvz = self.world_to_pixel_3d(points_xyz)
        uvz = tf.squeeze(uvz, axis=[0, 1])
        return uvz

    def world_to_pixel_2d(self, coords_xyz):
        tf.assert_rank(coords_xyz, 2)
        points_xyz = coords_xyz[tf.newaxis, ...]
        uvz = self.world_to_pixel_3d(points_xyz)
        uvz = tf.squeeze(uvz, axis=0)
        return uvz

    def world_to_pixel_3d(self, points_xyz):
        tf.assert_rank(points_xyz, 3)
        points_xyz = tf.cast(points_xyz, tf.float32)

        # Add ones for all points
        points_shape = tf.shape(points_xyz)[:2]
        new_shape = tf.concat([points_shape, [1]], axis=-1)
        points = tf.concat([points_xyz, tf.ones(new_shape, dtype=points_xyz.dtype)], axis=-1)
        # Project onto image plane
        projected_points = tf.matmul(self.projection_matrix, points, transpose_b=True)
        projected_points = tf.transpose(projected_points, [0, 2, 1])[..., :3]

        # Devide by Z
        uv = projected_points[..., :2] / projected_points[..., 2:3]
        uvz = tf.concat([uv, projected_points[..., 2:3]], axis=-1)
        return uvz

    def pixel_to_world(self, coords_uvz):
        if tf.rank(coords_uvz) == 1:
            points_uvz = coords_uvz[tf.newaxis, tf.newaxis, ...]
        elif tf.rank(coords_uvz) == 2:
            points_uvz = coords_uvz[tf.newaxis, ...]
        else:
            points_uvz = coords_uvz

        xyz = self.pixel_to_world_3d(points_uvz)

        if tf.rank(coords_uvz) == 1:
            xyz = tf.squeeze(xyz, axis=[0, 1])
        elif tf.rank(coords_uvz) == 2:
            xyz = tf.squeeze(xyz, axis=0)
        return xyz

    def pixel_to_world_1d(self, coords_uvz):
        tf.assert_rank(coords_uvz, 1)
        points_uvz = coords_uvz[tf.newaxis, tf.newaxis, ...]
        xyz = self.pixel_to_world_3d(points_uvz)
        xyz = tf.squeeze(xyz, axis=[0, 1])
        return xyz

    def pixel_to_world_2d(self, coords_uvz):
        tf.assert_rank(coords_uvz, 2)
        points_uvz = coords_uvz[tf.newaxis, ...]
        xyz = self.pixel_to_world_3d(points_uvz)
        xyz = tf.squeeze(xyz, axis=0)
        return xyz

    def pixel_to_world_3d(self, coords_uvz):
        tf.assert_rank(coords_uvz, 3)
        points_uvz = tf.cast(coords_uvz, tf.float32)
        multiplied_uv = points_uvz[..., 0:2] * points_uvz[..., 2:3]
        points_shape = tf.shape(points_uvz)[:2]
        new_shape = tf.concat([points_shape, [1]], axis=-1)
        ones = tf.ones(new_shape, dtype=points_uvz.dtype)
        multiplied_uvz1 = tf.concat([multiplied_uv, points_uvz[..., 2:3], ones], axis=-1)
        tranposed_xyz = tf.matmul(self.invr_projection_matrix, multiplied_uvz1, transpose_b=True)
        xyz = tf.transpose(tranposed_xyz, [0, 2, 1])[..., :3]
        return xyz


class CameraBighand(Camera):

    def __init__(self):
        super().__init__(focal_length=(475.065948, 475.065857),
                         principal_point=(315.944855, 245.287079),
                         image_size=(640, 480),
                         depth_unit=0.001,
                         )
        # self.focal_length = [476.0068, 476.0068]  # [588.235, 587.084]
        # extrinsic_matrix =
        # np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 0],
        #           [0.00469115935266, 0.999985218048, -0.00273845880292, 0],
        #           [-0.000969709653873, 0.00274303671904, 0.99999576807, 0],
        #           [0, 0, 0, 1]])


class CameraSR305(Camera):
    def __init__(self):
        super().__init__(focal_length=(476.0068, 476.0068),
                         principal_point=(313.6830139, 242.7547302),
                         image_size=(640, 480),
                         depth_unit=0.000125)


class CameraD105(Camera):
    def __init__(self):
        super().__init__(focal_length=(592.138, 592.138),
                         principal_point=(313.79, 238.076),
                         image_size=(640, 480),
                         depth_unit=0.001)


class CameraMSRA(Camera):
    def __init__(self):
        super().__init__(focal_length=(241.42, 241.42),
                         principal_point=(160, 120),
                         image_size=(320, 240))


def get_camera(camera_type: str) -> Camera:
    camera_type_lower = camera_type.lower()
    if camera_type_lower == 'sr305':
        return CameraSR305()
    if camera_type_lower == 'd105':
        return CameraD105()
    if camera_type_lower == 'msra':
        return CameraMSRA()
    if camera_type_lower == 'bighand':
        return CameraBighand()
    raise RuntimeError(f"Invalid camera type: {camera_type}")
