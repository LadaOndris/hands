import numpy as np
from PIL import Image

from src.utils.paths import NYU_DATASET_DIR
from src.utils.plots import plot_depth_image


def plot_sample_images(samples: int) -> None:
    for i in range(samples):
        im = np.array(Image.open(NYU_DATASET_DIR.joinpath(F"train/depth_1_000{i}001.png")))

        # The top 8 bits of depth are packed into green and the lower 8 bits into blue.
        # RGB
        green = im[:, :, 1].astype(np.uint16)
        blue = im[:, :, 2].astype(np.uint16)
        depth_im = np.left_shift(green, 8) + blue
        print("{}, max={}, min={}, mean={}".format(
            depth_im.shape, np.max(depth_im), np.min(depth_im), np.mean(depth_im)))
        plot_depth_image(depth_im)



if __name__ == "__main__":
    plot_sample_images(samples=10)
