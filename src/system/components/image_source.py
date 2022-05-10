from typing import List

import numpy as np
import pyrealsense2 as rs

from postoperations.calibration.calibrate import extrinsics_from_rotation_and_translation
from src.system.components.base import ImageSource
from src.utils.imaging import crop_to_equal_dims
from src.utils.live import get_depth_unit


class RealSenseCameraWrapper:

    def __init__(self, enable_depth: bool, enable_color: bool):
        if not enable_color and not enable_depth:
            raise ValueError('Both color and depth cannot be disabled.')
        self.is_depth_enabled = enable_depth
        self.is_color_enabled = enable_color
        self.preset_name = "Default"

        self.pipeline, self.profile = self.start_pipeline()

        self.depth_image_source: ImageSource = DepthRealSenseImageSource(self.pipeline, self.profile)
        self.color_image_source: ImageSource = ColorRealSenseImageSource(self.pipeline, self.profile)

    def start_pipeline(self) -> (rs.pipeline, rs.pipeline_profile):
        pipeline = rs.pipeline()
        config = rs.config()
        if self.is_depth_enabled:
            config.enable_stream(rs.stream.depth, 640, 480)
        if self.is_color_enabled:
            config.enable_stream(rs.stream.color, 640, 480)
        profile = pipeline.start(config)

        if self.is_depth_enabled:
            # Set Hand preset
            depth_sensor = profile.get_device().first_depth_sensor()

            # Set preset
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            print('preset range:' + str(preset_range))
            for i in range(int(preset_range.max)):
                visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
                print('%02d: %s' % (i, visulpreset))
                if visulpreset == self.preset_name:
                    depth_sensor.set_option(rs.option.visual_preset, i)
                    break

            # if camera is not SR305, which does not support the following settings
            camera_name = profile.get_device().get_info(rs.camera_info.name)
            if "SR305" not in camera_name:
                # Set disparity shift
                device = rs.context().query_devices()[0]
                advnc_mode = rs.rs400_advanced_mode(device)
                depth_table_control_group = advnc_mode.get_depth_table()
                depth_table_control_group.disparityShift = 0
                advnc_mode.set_depth_table(depth_table_control_group)
                rsm = advnc_mode.get_rsm()
                rsm.removeThresh = 100
                advnc_mode.set_rsm(rsm)

                # Set auto exposure to avoid overexposure or underexposure
                depth_sensor.set_option(rs.option.enable_auto_exposure, True)

                # Set maximum laser power
                laser_range = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, laser_range.max)



        return pipeline, profile

    def get_color_image_source(self) -> ImageSource:
        if not self.is_color_enabled:
            raise RuntimeError('Color stream was not enabled!')
        return self.color_image_source

    def get_depth_image_source(self) -> ImageSource:
        if not self.is_depth_enabled:
            raise RuntimeError('Depth stream was not enabled!')
        return self.depth_image_source

    def get_depth_intrinsics(self) -> np.ndarray:
        intrinsics = self._get_stream_intrinsics(rs.stream.depth)
        return self._realsense_intrinsics_to_matrix(intrinsics)

    def get_color_intrinsics(self) -> np.ndarray:
        intrinsics = self._get_stream_intrinsics(rs.stream.color)
        return self._realsense_intrinsics_to_matrix(intrinsics)

    def _get_stream_intrinsics(self, stream: rs.stream):
        stream = self._get_stream(stream)
        intrinsics = stream.get_intrinsics()
        return intrinsics

    def _get_stream(self, stream: rs.stream):
        return self.profile.get_stream(stream).as_video_stream_profile()

    def _realsense_intrinsics_to_matrix(self, intrinsics) -> np.ndarray:
        fx = intrinsics.fx
        fy = intrinsics.fy
        ppx = intrinsics.ppx
        ppy = intrinsics.ppy
        return np.array([[fx, 0, ppx],
                         [0, fy, ppy],
                         [0, 0, 1]])

    def get_depth_to_color_extrinsics(self) -> np.ndarray:
        depth_stream = self._get_stream(rs.stream.depth)
        color_stream = self._get_stream(rs.stream.color)
        depth_to_color_extrin = depth_stream.get_extrinsics_to(color_stream)
        return self._realsense_extrinsics_to_matrix(depth_to_color_extrin)

    def _realsense_extrinsics_to_matrix(self, extrinsics) -> np.ndarray:
        rot_mat = np.asarray(extrinsics.rotation).reshape([3, 3])
        tran_vec = np.asarray(extrinsics.translation)
        extr = extrinsics_from_rotation_and_translation(rot_mat, tran_vec)
        return extr

    def get_depth_unit(self):
        return get_depth_unit(self.profile)

    def __del__(self):
        if hasattr(self, "pipeline"):
            self.pipeline.stop()


class ColorRealSenseImageSource(ImageSource):

    def __init__(self, pipeline: rs.pipeline, profile: rs.pipeline_profile):
        self.pipeline = pipeline
        self.profile = profile
        self.previous_image = None

    def get_new_image(self):
        frameset = self.pipeline.wait_for_frames()
        frame = frameset.get_color_frame()
        image = np.asarray(frame.get_data())
        self.previous_image = image
        return image

    def get_previous_image(self):
        return self.previous_image


class DepthRealSenseImageSource(ImageSource):

    def __init__(self, pipeline: rs.pipeline, profile: rs.pipeline_profile, max_depth=1000):
        """
        Captures image from an Intel RealSense camera.

        Parameters
        ----------
        max_depth
            Max depth value in millimeters.
            Values bigger than max_depth are replace with the depth of background.
        """
        self.pipeline = pipeline
        self.profile = profile
        self.max_depth = max_depth
        self.previous_image = None

        depth_unit = get_depth_unit(profile)
        millimeter = 0.001
        self.depth_unit_correction_factor = depth_unit / millimeter

    def get_new_image(self, filters: List[rs.filter] = None):
        frameset = self.pipeline.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        if filters is not None:
            for filter in filters:
                depth_frame = filter.process(depth_frame)
        depth_image = np.array(depth_frame.get_data())
        depth_image = depth_image[..., np.newaxis] * self.depth_unit_correction_factor
        # Remove background further than max_depth mm.
        depth_image[depth_image > self.max_depth] = 0
        # Convert float64 back to uint16.
        depth_image = depth_image.astype(np.uint16)
        depth_image = crop_to_equal_dims(depth_image)
        self.previous_image = depth_image
        return depth_image

    def get_previous_image(self):
        return self.previous_image
