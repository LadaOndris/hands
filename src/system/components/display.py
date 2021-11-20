import cv2

from src.system.components.base import Display
from src.system.components.image_source import LiveRealSenseImageSource


class OpencvDisplay(Display):

    def __init__(self):
        self.window_same = 'window'
        cv2.namedWindow(self.window_same, cv2.WINDOW_NORMAL)

    def update(self, image, keypoints=None):
        # Convert depth image to something cv2 can understand
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.4), cv2.COLORMAP_BONE)

        cv2.imshow(self.window_same, depth_colormap)
        # Don't wait for the user to press a key
        cv2.waitKey(1)


if __name__ == "__main__":
    display = OpencvDisplay()
    img_source = LiveRealSenseImageSource()
    while True:
        img = img_source.next_image()
        display.update(img)
