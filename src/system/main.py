import time

from src.system.components import coordinate_predictors
from src.system.components.base import Display, GestureRecognizer, ImageSource


class System:
    """
    LiveGestureRecognizer reads image from image source, predicts hand keypoints,
    recognizes gesture using given GestureRecognizer, and displays the result.
    """

    def __init__(self, image_source: ImageSource, predictor: coordinate_predictors,
                 display: Display, gesture_recognizer: GestureRecognizer = None,
                 measure_time: bool = False):
        self.image_source = image_source
        self.predictor = predictor
        self.display = display
        self.gesture_recognizer = gesture_recognizer
        self.measure_time = measure_time

    def start(self):
        while True:
            # Capture image to process next
            if self.measure_time:
                start_time = time.time()

            image = self.image_source.get_new_image()
            if image is None:
                break
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
            if self.measure_time:
                print("Elapsed time [ms]: {:.0f}".format((time.time() - start_time) * 1000))
