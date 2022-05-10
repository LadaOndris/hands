import logging

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.acceptance.gesture_acceptance_result import GestureAcceptanceResult
from src.system.components.base import GestureRecognizer
from src.system.database.reader import UsecaseDatabaseReader


def get_trained_model(gestures_folder: str):
    reader = UsecaseDatabaseReader()
    reader.load_from_subdir(gestures_folder)

    x = reader.hand_poses
    y = reader.labels

    x_flattened = np.reshape(x, [x.shape[0], -1])
    y = y.astype(int)

    # classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
    # classifier = DecisionTreeClassifier(max_depth=5)
    classifier = MLPClassifier(max_iter=1000)
    classifier = classifier.fit(x_flattened, y)
    logging.info("Gesture recognition score: ", classifier.score(x_flattened, y))
    return classifier


class ClassifierGestureRecognizer(GestureRecognizer):

    def __init__(self, gestures_folder: str):
        self.gesture_model = get_trained_model(gestures_folder)

    def recognize(self, keypoints_xyz: np.ndarray) -> GestureAcceptanceResult:
        keypoints_flattened = np.reshape(keypoints_xyz, [1, -1])
        prediction = self.gesture_model.predict(keypoints_flattened)[0]
        prediction = round(prediction)

        result = GestureAcceptanceResult()
        result.gesture_label = str(prediction)
        result.is_gesture_valid = True
        return result
