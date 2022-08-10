from typing import List

import cv2
import mediapipe as mp
import numpy as np

from src.system.components.base import CoordinatePrediction, CoordinatePredictor

mp_hands = mp.solutions.hands


class MediapipeCoordinatePredictor(CoordinatePredictor):

    def __init__(self):
        super().__init__()
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def predict(self, image: np.ndarray) -> CoordinatePrediction:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        if results.multi_hand_landmarks is None:
            return None

        world_coordinates = self._mediapipe_landmarks_to_ndarray(
            results.multi_hand_world_landmarks)

        image_coordinates = self._mediapipe_landmarks_to_ndarray(
            results.multi_hand_landmarks)

        # Denormalize
        image_coordinates[:, 0] *= image.shape[1]
        image_coordinates[:, 1] *= image.shape[0]
        image_coordinates = image_coordinates.astype(int)

        return CoordinatePrediction(world_coordinates, image_coordinates)

    def _mediapipe_landmarks_to_ndarray(self, landmarks: List) -> np.ndarray:
        array = np.empty((21, 3), dtype=float)
        for index, landmark in enumerate(landmarks[0].landmark):
            array[index, 0] = landmark.x
            array[index, 1] = landmark.y
            array[index, 2] = landmark.z
        return array

    def __exit__(self):
        self.hands.close()
