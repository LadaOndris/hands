import logging

import numpy as np
from sklearn.neural_network import MLPClassifier

from gestures.relative_distance.base import _upper_tri, scaled_distance_matrix
from src.gestures.gesture_acceptance_result import GestureRecognitionResult
from src.system.components.base import GestureRecognizer
from src.system.database.reader import UsecaseDatabaseReader


def prepare_features(x: np.ndarray) -> np.ndarray:
    # The features are relative distances
    distance_matrix = scaled_distance_matrix(x)
    relative_distances = _upper_tri(distance_matrix)
    return relative_distances


def get_trained_model(gestures_folder: str):
    reader = UsecaseDatabaseReader()
    reader.load_from_subdir(gestures_folder)

    x = reader.hand_poses
    y = reader.labels

    features = prepare_features(x)
    features_flattened = np.reshape(features, [features.shape[0], -1])
    y = y.astype(int)

    # classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
    # classifier = DecisionTreeClassifier(max_depth=5)
    classifier = MLPClassifier(max_iter=1000)
    classifier = classifier.fit(features_flattened, y)
    logging.info("Gesture recognition score: ", classifier.score(features_flattened, y))
    return classifier


class MLPGestureRecognizer(GestureRecognizer):
    """
    Uses a trained MLP classifier to recognize gestures.
    It requires a database of representative gestures to train on.
    """

    def __init__(self, gestures_folder: str):
        self.gesture_model = get_trained_model(gestures_folder)

    def recognize(self, keypoints_xyz: np.ndarray) -> GestureRecognitionResult:
        features = prepare_features(keypoints_xyz[np.newaxis, ...])
        features_flattened = np.reshape(features, [1, -1])
        prediction = self.gesture_model.predict(features_flattened)[0]
        prediction = round(prediction)

        result = GestureRecognitionResult()
        result.gesture_label = str(prediction)
        result.is_gesture_valid = True
        return result
