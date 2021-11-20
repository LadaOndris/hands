import argparse

from system.components.base import Detector, Display, Estimator, ImageSource, KeypointsToRectangle
from system.components.detector import BlazehandDetector
from system.components.display import OpencvDisplay
from system.components.estimator import BlazeposeEstimator
from system.components.image_source import LiveRealSenseImageSource
from system.components.keypoints_to_rectangle import KeypointsToRectangleImpl


class HandTracker:
    """
    HandTracker controls the logic behind detecting and estimating.
    Estimation is performed only if there is a detected hand, and
    detection is not enabled unless there is no hand to being tracked.
    """

    def __init__(self, image_source: ImageSource, detector: Detector, estimator: Estimator,
                 keypoints_to_rectangle: KeypointsToRectangle, display: Display):
        self.image_source = image_source
        self.detector = detector
        self.estimator = estimator
        self.keypoints_to_rectangle = keypoints_to_rectangle
        self.display = display

    def track(self):
        rectangle = None
        keypoints = None

        while True:
            # Capture image to process next
            image = self.image_source.next_image()

            # Detect when there no hand being tracked
            if keypoints is None:
                rectangle = self.detector.detect(image)

            # Estimate keypoints if there a hand detected
            if rectangle is not None:
                reject_hand_flag, keypoints = self.estimator.estimate(image, rectangle)

                # Reject if the hand is not present
                if reject_hand_flag:
                    keypoints = None
                # Display the predicted keypoints and prepare for the next frame
                else:
                    self.display.update(image, keypoints)
                    keypoints = keypoints
                    rectangle = self.keypoints_to_rectangle.convert(keypoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    tracker = HandTracker(image_source=LiveRealSenseImageSource(),
                          detector=BlazehandDetector(),
                          estimator=BlazeposeEstimator(),
                          keypoints_to_rectangle=KeypointsToRectangleImpl(),
                          display=OpencvDisplay())
    tracker.track()

