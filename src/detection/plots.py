import tensorflow as tf
from matplotlib import patches as patches, pyplot as plt

from src.utils.plots import plotlive, save_show_fig

depth_image_cmap = 'gist_yarg'
prediction_box_color = '#B73229'
blue_color = '#293e65'
boxes_color = '#141d32'


def draw_bboxes(imgs, bboxes):
    """Drawing bounding boxes on given images.
    inputs:
        imgs = (batch_size, height, width, channels)
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    colors = tf.constant([[1, 0, 0]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)

    for img_with_bb in imgs_with_bb:
        fig, ax = image_plot()
        ax.imshow(img_with_bb, cmap=depth_image_cmap)
        save_show_fig(fig, None, True)


def plot_predictions(image, boxes, fig_location):
    fig, ax = image_plot()
    _plot_predictions(fig, ax, image, boxes)
    save_show_fig(fig, fig_location, True)


@plotlive
def plot_predictions_live(fig, ax, image, boxes):
    _plot_predictions(fig, ax, image, boxes)


def _plot_predictions(fig, ax, image, boxes):
    ax.imshow(image, cmap=depth_image_cmap)
    if tf.is_tensor(boxes):
        boxes = boxes.numpy()
    for i in range(boxes.shape[0]):
        x, y = boxes[i, 0:2]
        w, h = (boxes[i, 2:4] - boxes[i, 0:2])
        plot_prediction_box(ax, x, y, w, h)

    plot_adjust(fig, ax)


def plot_adjust(fig, ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)


def plot_predictions_with_cells(image, boxes, nums, stride, fig_location=None):
    fig, ax = image_plot()
    ax.imshow(image, cmap=depth_image_cmap)

    for i in range(nums):
        x, y = boxes[i, 0:2].numpy()
        w, h = (boxes[i, 2:4] - boxes[i, 0:2]).numpy()

        centroid = (x + w / 2, y + h / 2)
        plot_prediction_box(ax, x, y, w, h)
        plot_responsible_cell(ax, centroid, stride)
        plot_centroid(ax, centroid)

    plot_adjust(fig, ax)
    save_show_fig(fig, fig_location, True)


def plot_prediction_box(ax, x, y, w, h):
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=prediction_box_color, facecolor='none')
    ax.add_patch(rect)


def plot_centroid(ax, centroid):
    centroid = patches.Circle((centroid[0], centroid[1]), radius=4, facecolor=prediction_box_color)
    ax.add_patch(centroid)


def plot_responsible_cell(ax, centroid, stride):
    rect = patches.Rectangle((centroid[0] // stride * stride, centroid[1] // stride * stride), stride, stride,
                             linewidth=2, edgecolor=blue_color, facecolor='none')
    ax.add_patch(rect)


def image_plot():
    return plt.subplots(1, figsize=(4, 4))
