import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import ImageDraw


def draw_bboxes(imgs, bboxes):
    """Drawing bounding boxes on given images.
    inputs:
        imgs = (batch_size, height, width, channels)
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    colors = tf.constant([[1, 0, 0]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()

#
# def darw_bboxes(img, bboxes):
#     """Drawing bounding boxes on given image.
#     inputs:
#         img = (height, width, channels)
#         bboxes = (total_bboxes, [y1, x1, y2, x2])
#     """
#     image = tf.keras.preprocessing.image.array_to_img(img)
#     width, height = image.size
#     draw = ImageDraw.Draw(image)
#     color = (255, 0, 0, 255)
#     for index, bbox in enumerate(bboxes):
#         y1, x1, y2, x2 = tf.split(bbox, 4)
#         width = x2 - x1
#         height = y2 - y1
#         if width <= 0 or height <= 0:
#             continue
#         draw.rectangle((x1, y1, x2, y2), outline=color, width=1)
#     plt.figure()
#     plt.imshow(image)
#     plt.show()
