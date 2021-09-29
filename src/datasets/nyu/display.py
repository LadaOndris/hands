import numpy as np
from PIL import Image

from src.utils.paths import NYU_DATASET_DIR
from src.utils.plots import plot_depth_image

im = np.array(Image.open(NYU_DATASET_DIR.joinpath('train/depth_1_0001001.png')))

# The top 8 bits of depth are packed into green and the lower 8 bits into blue.
# RGB
green = im[:, :, 1].astype(np.uint16)
blue = im[:, :, 2].astype(np.uint16)
depth_im = np.left_shift(green, 8) + blue
print(depth_im.shape, np.max(depth_im), np.min(depth_im), np.mean(depth_im))
plot_depth_image(depth_im)
