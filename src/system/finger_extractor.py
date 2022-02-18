import cv2 as cv
import numpy as np

from src.postoperations.extraction import ExtractedFingersDisplay, FingerExtractor
from src.utils.camera import CameraBighand
from system.components.detector import BlazehandDetector
from system.components.display import EmptyDisplay, OpencvDisplay
from system.components.estimator import BlazeposeEstimator
from system.components.image_source import RealSenseCameraWrapper
from system.components.keypoints_to_rectangle import KeypointsToRectangleImpl
from system.tracking import HandTracker


class CoordsConvertor:

    def align_to_other_stream(self, hulls):
        # The original 640 image is cropped to 480.
        # The infrared stays at 640, so the x coordinate needs to be shifted.
        for hull in hulls:
            hull[..., 0] += 80
        return hulls


if __name__ == "__main__":
    realsense_wrapper = RealSenseCameraWrapper(enable_depth=True, enable_color=True)
    color_image_source = realsense_wrapper.get_color_image_source()
    depth_image_source = realsense_wrapper.get_depth_image_source()
    estimator = BlazeposeEstimator(CameraBighand())
    hand_tracker = HandTracker(image_source=realsense_wrapper.get_depth_image_source(),
                               detector=BlazehandDetector(),
                               estimator=estimator,
                               keypoints_to_rectangle=KeypointsToRectangleImpl(),
                               display=OpencvDisplay())
    finger_extractor = FingerExtractor()
    coords_converter = CoordsConvertor()
    extraction_display = ExtractedFingersDisplay()

    for keypoints, norm_keypoints, normalized_image, crop_offset in hand_tracker.track():
        # TODO: is the hand at the distance of 40 cm or closer?
        # Track the hand if not.

        # TODO: accept gesture?
        is_gesture_accepted = True

        depth_image = depth_image_source.get_new_image()
        color_image = color_image_source.get_new_image()

        hulls_in_cropped = finger_extractor.extract_hulls(normalized_image, norm_keypoints * 256)
        # TODO: postprocesses normalized coordinates to coordinates
        # of the originally captured image
        if len(hulls_in_cropped) > 0:
            hulls = []
            for hull_in_cropped in hulls_in_cropped:
                hull = estimator.postprocess(hull_in_cropped.squeeze() / 256, crop_offset).numpy().astype(int)
                hulls.append(hull[:, np.newaxis, :])

            aligned_hulls = coords_converter.align_to_other_stream(hulls)

            extraction_display.display(color_image, aligned_hulls)
