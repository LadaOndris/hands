import time

from src.system.components import CoordinatePredictor
from src.system.components.base import Display, GestureRecognizer, ImageSource


class LiveGestureRecognizer:
    """
    LiveGestureRecognizer reads image from image source, predicts hand keypoints,
    recognizes gesture using given GestureRecognizer, and plots the result
    in a window.
    """

    def __init__(self, image_source: ImageSource, predictor: CoordinatePredictor,
                 display: Display, gesture_recognizer: GestureRecognizer = None):
        self.image_source = image_source
        self.predictor = predictor
        self.display = display
        self.gesture_recognizer = gesture_recognizer

    def start(self):
        while True:
            # Capture image to process next
            start_time = time.time()
            image = self.image_source.get_new_image()

            prediction = self.predictor.predict(image)
            if prediction is None:
                self.display.update(image)
            else:
                gesture_label = None
                if self.gesture_recognizer is not None:
                    gesture_result = self.gesture_recognizer.recognize(prediction.world_coordinates)
                    if gesture_result.is_gesture_valid:
                        gesture_label = gesture_result.gesture_label
                self.display.update(image, keypoints=prediction.image_coordinates,
                                    gesture_label=gesture_label)
            print("Elapsed time [ms]: {:.0f}".format((time.time() - start_time) * 1000))
