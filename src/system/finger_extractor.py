import cv2 as cv

from src.postoperations.extraction import ExtractedFingersDisplay, FingerExtractor
from src.utils.camera import CameraBighand
from system.components.detector import BlazehandDetector
from system.components.display import EmptyDisplay
from system.components.estimator import BlazeposeEstimator
from system.components.image_source import RealSenseCameraWrapper
from system.components.keypoints_to_rectangle import KeypointsToRectangleImpl
from system.tracking import HandTracker

class CoordsConvertor:

    def align_to_other_stream(self, coords):
        return coords

if __name__ == "__main__":
    realsense_wrapper = RealSenseCameraWrapper(enable_depth=True, enable_color=False)
    color_image_source = realsense_wrapper.get_color_image_source()
    depth_image_source = realsense_wrapper.get_depth_image_source()
    hand_tracker = HandTracker(image_source=realsense_wrapper.get_depth_image_source(),
                               detector=BlazehandDetector(),
                               estimator=BlazeposeEstimator(CameraBighand()),
                               keypoints_to_rectangle=KeypointsToRectangleImpl(),
                               display=EmptyDisplay())
    finger_extractor = FingerExtractor()
    coords_converter = CoordsConvertor()
    extraction_display = ExtractedFingersDisplay()

    for keypoints in hand_tracker.track():  # TODO: should yield coordinates in range of image size
        # TODO: accept gesture?
        is_gesture_accepted = True

        depth_image = depth_image_source.get_new_image()
        color_image = color_image_source.get_new_image()

        hulls = finger_extractor.extract_hulls(depth_image, keypoints)
        aligned_hulls = coords_converter.align_to_other_stream(hulls)

        extraction_display.display(color_image, aligned_hulls)
