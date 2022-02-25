from typing import List

import numpy as np
import cv2 as cv
from postoperations.calibration.calibrate import deproject_pixels, extrinsics, project_points, transform_points_to_points
from src.postoperations.extraction import ExtractedFingersDisplay, FingerExtractor
from src.utils.camera import CameraBighand
from system.components.detector import BlazehandDetector
from system.components.display import OpencvDisplay
from system.components.estimator import BlazeposeEstimator
from system.components.image_source import RealSenseCameraWrapper
from system.components.keypoints_to_rectangle import KeypointsToRectangleImpl
from system.tracking import HandTracker


class CoordsConvertor:
    """
    Performes the conversion between two cameras' coordinate systems.
    Required information are intrinsic and extrinsic parameters.
    """

    def __init__(self, depth_intrinsics, color_intrinsics, extrinsics):
        self.depth_intrinsics = depth_intrinsics
        self.color_intrinsics = color_intrinsics
        self.extrinsics = extrinsics

    def align_to_other_stream(self, hulls):
        for i, hull in enumerate(hulls):
            depth_points = deproject_pixels(hull, self.depth_intrinsics)
            color_points = transform_points_to_points(depth_points, self.extrinsics)
            color_pixels = project_points(color_points, self.color_intrinsics)
            hulls[i] = color_pixels.astype(int)
        return hulls

def get_contour_depth(depth_image, contour):
    # Create each finger's mask
    mask = np.zeros(depth_image.shape, np.uint8)
    ALL_CONTOURS = -1
    FILLED = -1
    cv.drawContours(mask, [contour], ALL_CONTOURS, (255,), FILLED)

    depth_values = depth_image[np.nonzero(mask)]
    mean_depth = np.mean(depth_values[np.nonzero(depth_values)])
    return mean_depth

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
    coords_converter = CoordsConvertor(realsense_wrapper.get_depth_intrinsics(),
                                       realsense_wrapper.get_color_intrinsics(),
                                       #realsense_wrapper.get_depth_to_color_extrinsics())
                                       extrinsics=extrinsics())
    extraction_display = ExtractedFingersDisplay()

    for keypoints, norm_keypoints, normalized_image, crop_offset in hand_tracker.track():
        # TODO: is the hand at the distance of 40 cm or closer?
        # Track the hand if not.

        # TODO: accept gesture?
        is_gesture_accepted = True

        # Also convert the depth values to meters. get_previous_image() returns pixels in mm.
        previous_depth_image = depth_image_source.get_previous_image() * 0.001
        depth_image = depth_image_source.get_new_image()
        color_image = color_image_source.get_new_image()

        hulls_in_cropped, contours_in_cropped = finger_extractor.extract_hulls(normalized_image, norm_keypoints * estimator.model_image_size)
        if len(hulls_in_cropped) > 0:
            # hulls = []
            # for hull_in_cropped in hulls_in_cropped:
            #     hull_coords_normalized = hull_in_cropped.squeeze() / estimator.model_image_size
            #     hull = estimator.postprocess(hull_coords_normalized, crop_offset).numpy().astype(int)
            #     hull[..., 0] += 80
            #     hulls.append(hull)

            # TODO: Do not fill the whole contour, but only the fingertip
            # Fill contours here to create finger masks and retrieve depth of each finger
            contours_with_depth = []
            for contour_in_cropped in contours_in_cropped:
                contour_coords_normalized = contour_in_cropped.squeeze() / estimator.model_image_size
                contour = estimator.postprocess(contour_coords_normalized, crop_offset).numpy().astype(int)
                contour_mean_depth = get_contour_depth(previous_depth_image, contour)
                contour_coords_depth = np.full(shape=(contour.shape[0], 1), fill_value=contour_mean_depth)
                contour_with_depth = np.concatenate([contour, contour_coords_depth], axis=-1)
                contour_with_depth[..., 0] += 80
                contours_with_depth.append(contour_with_depth)

            # Requires image in meters! Intrinsics and extrinsics are in meters as well!
            aligned_hulls = coords_converter.align_to_other_stream(contours_with_depth)

            extraction_display.display(color_image, aligned_hulls)
