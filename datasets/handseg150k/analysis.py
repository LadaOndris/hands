from PIL import Image
from matplotlib import pyplot as plt
from collections import Counter
from pathlib import Path
import numpy as np
import os

dirname = os.path.dirname(__file__)
    

depth_im = np.array(Image.open(os.path.join(dirname, 'images/user-4.00000003.png')))# Loading depth image
mask_im = np.array(Image.open(os.path.join(dirname, 'masks/user-4.00000003.png')))#  Loading mask image
depth_im = depth_im.astype(np.float32)# Converting to float
mean = np.mean(depth_im)
print(Counter(depth_im.flatten()))
#plt.hist(mask_im, bins=100)
#plt.show()
mean_depth_ims = 10000.0 # Mean value of the depth images
depth_im /= mean_depth_ims # Normalizing depth image
plt.imshow(depth_im); plt.title('Depth Image'); plt.show() # Displaying Depth Image
plt.imshow(mask_im); plt.title('Mask Image'); plt.show() # Displaying Mask Image


print(depth_im.shape)
def get_image():
    return depth_im, mask_im