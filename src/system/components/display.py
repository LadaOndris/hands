import cv2

from src.system.components.base import Display
from src.system.components.image_source import RealSenseCameraWrapper


class OpencvDisplay(Display):

    def __init__(self):
        self.window_same = 'window'
        cv2.namedWindow(self.window_same, cv2.WINDOW_NORMAL)

    def update(self, image, keypoints=None, bounding_boxes=None):
        # Convert depth image to something cv2 can understand
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.4), cv2.COLORMAP_BONE)

        # Draw rectangles in the depth colormap image
        if bounding_boxes is not None:
            for rect in bounding_boxes:
                cv2.rectangle(depth_colormap, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
        if keypoints is not None:
            for point in keypoints:
                cv2.circle(depth_colormap, (point[0], point[1]), radius=0, color=(0, 0, 255), thickness=4)

        cv2.imshow(self.window_same, depth_colormap)
        # Don't wait for the user to press a key
        cv2.waitKey(1)


if __name__ == "__main__":
    display = OpencvDisplay()
    realsense_wrapper = RealSenseCameraWrapper()
    img_source = realsense_wrapper.get_depth_image_source()
    while True:
        img = img_source.get_new_image()
        display.update(img)
