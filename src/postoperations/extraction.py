"""
Finger extraction
"""

import numpy as np

from src.utils.paths import OTHER_DIR
from src.utils.plots import plot_image_with_skeleton

datetime = F"20211213-141927"
img_path = OTHER_DIR.joinpath(F"extraction/{datetime}_image.npy")
jnt_path = OTHER_DIR.joinpath(F"extraction/{datetime}_joints.npy")

img = np.load(img_path)
jnt = np.load(jnt_path)



plot_image_with_skeleton(img, jnt[:, :2] * 255)
