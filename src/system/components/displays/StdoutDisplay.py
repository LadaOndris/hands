from src.system.components.base import Display


class StdoutDisplay(Display):

    def update(self, image, keypoints=None, bounding_boxes=None, gesture_label: str = None):
        if gesture_label is None:
            gesture_label = "-"
        print(f'Gesture: {gesture_label}')