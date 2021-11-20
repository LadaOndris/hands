import numpy as np
import pyrealsense2 as rs

from src.utils.live import get_depth_unit
from system.components.base import ImageSource


class LiveRealSenseImageSource(ImageSource):

    def __init__(self, max_depth=1000):
        """
        Captures image from an Intel RealSense camera.

        Parameters
        ----------
        max_depth
            Max depth value in millimeters.
            Values bigger than max_depth are replace with the depth of background.
        """
        self.max_depth = max_depth
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480)
        profile = self.pipe.start(cfg)
        depth_unit = get_depth_unit(profile)  # SR305 returns 0.00012498 (mm)
        millimeter = 0.001
        self.depth_unit_correction_factor = depth_unit / millimeter

    def next_image(self):
        frameset = self.pipe.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        depth_image = np.array(depth_frame.get_data())
        depth_image = depth_image[..., np.newaxis] * self.depth_unit_correction_factor
        # Remove background further than max_depth mm.
        depth_image[depth_image > self.max_depth] = 0
        # Convert float64 back to uint16.
        depth_image = depth_image.astype(np.uint16)
        return depth_image

    def __del__(self):
        self.pipe.stop()