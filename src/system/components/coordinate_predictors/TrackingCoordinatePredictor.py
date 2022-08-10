import numpy as np
import tensorflow as tf

from src.system.components.base import CoordinatePrediction, CoordinatePredictor, Detector, Estimator, KeypointsToRectangle
from src.utils.camera import Camera


class TrackingCoordinatePredictor(CoordinatePredictor):
    """
    TrackingCoordinatePredictor combines hand detection and pose estimation into a single system.
    Estimation is performed only if there is a detected hand, and
    detection is not enabled unless there is no hand being tracked.
    """

    def __init__(self, detector: Detector, estimator: Estimator,
                 keypoints_to_rectangle: KeypointsToRectangle, camera: Camera):
        super().__init__()
        self.detector = detector
        self.estimator = estimator
        self.keypoints_to_rectangle = keypoints_to_rectangle
        self.camera = camera

        self.rectangle = None
        self.keypoints = None

    def predict(self, image: np.ndarray) -> CoordinatePrediction:
        image = tf.convert_to_tensor(image)
        # Detect hand if none is being tracker
        if self.keypoints is None:
            self.rectangle = self.detector.detect(image)[0]

        # Estimate keypoints if there a hand detected
        if not tf.experimental.numpy.allclose(self.rectangle, 0):
            hand_presence_flag, keypoints_uvz, normalized_keypoints_uvz, normalized_image, crop_offset_uv = \
                self.estimator.estimate(image, self.rectangle)
            self.keypoints = keypoints_uvz
            if hand_presence_flag:
                self.rectangle = self.keypoints_to_rectangle.convert(keypoints_uvz)
                keypoints_xyz = self.camera.pixel_to_world(keypoints_uvz)
                return CoordinatePrediction(keypoints_xyz, keypoints_uvz)
        return None