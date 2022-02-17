import numpy as np
import pyrealsense2 as rs

from src.system.components.base import ImageSource
from src.utils.imaging import crop_to_equal_dims
from src.utils.live import get_depth_unit


class RealSenseCameraWrapper:

    def __init__(self, enable_depth: bool, enable_color: bool):
        if not enable_color and not enable_depth:
            raise ValueError('Both color and depth cannot be disabled.')
        self.is_depth_enabled = enable_depth
        self.is_color_enabled = enable_color

        self.pipeline, profile = self.start_pipeline()

        self.depth_image_source: ImageSource = DepthRealSenseImageSource(self.pipeline, profile)
        self.color_image_source: ImageSource = ColorRealSenseImageSource(self.pipeline, profile)

    def start_pipeline(self) -> (rs.pipeline, rs.pipeline_profile):
        pipeline = rs.pipeline()
        config = rs.config()
        if self.is_depth_enabled:
            config.enable_stream(rs.stream.depth, 640, 480)
        if self.is_color_enabled:
            config.enable_stream(rs.stream.color, 640, 480)
        profile = pipeline.start(config)
        return pipeline, profile

    def get_color_image_source(self) -> ImageSource:
        if not self.is_color_enabled:
            raise RuntimeError('Color stream was not enabled!')
        return self.color_image_source

    def get_depth_image_source(self) -> ImageSource:
        if not self.is_depth_enabled:
            raise RuntimeError('Depth stream was not enabled!')
        return self.depth_image_source

    def __del__(self):
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

    def get_new_image(self):
        frameset = self.pipeline.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
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
