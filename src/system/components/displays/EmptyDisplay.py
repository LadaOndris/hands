from src.system.components.base import Display


class EmptyDisplay(Display):

    def update(self, image, keypoints=None, bounding_boxes=None, gesture_label: str = None):
        pass