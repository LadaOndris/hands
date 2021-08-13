import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from src.detection.plots import image_plot, plot_prediction_box
from src.utils.camera import Camera
from src.utils.paths import BIGHAND_DATASET_DIR
from src.utils.plots import _plot_depth_image


def boxes_from_joints(joints):
    x = joints[..., 0]
    y = joints[..., 1]
    xmin = np.min(x, axis=-1)
    xmax = np.max(x, axis=-1)
    ymin = np.min(y, axis=-1)
    ymax = np.max(y, axis=-1)
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


def plot_palm_joints_and_box(img, joints, box):
    fig, ax = image_plot()
    _plot_depth_image(ax, img)
    plot_prediction_box(ax, box[0], box[1], box[2] - box[0], box[3] - box[1])
    ax.scatter(joints[..., 0], joints[..., 1])
    fig.show()


def create_for_file(source_path: str, save_path: str):
    camera = Camera('bighand')

    df = pd.read_csv(source_path, sep="\t", header=None)
    df_joints_xyz = df.iloc[:, 1:]
    np_joints_xyz = df_joints_xyz.to_numpy()
    np_joints_xyz = np.reshape(np_joints_xyz, [-1, 21, 3])
    tf_joints_xyz = tf.convert_to_tensor(np_joints_xyz)
    tf_joints_uvz = camera.world_to_pixel(tf_joints_xyz)
    np_joints_uv = tf_joints_uvz[..., :2].numpy()
    palm_joints = np_joints_uv[..., :6]
    boxes = boxes_from_joints(palm_joints)
    img = Image.open(os.path.join(BIGHAND_DATASET_DIR, df.iloc[10000, 0]))
    plot_palm_joints_and_box(img, palm_joints[0], boxes[0])
    pass


def annotation_files_for_subject(subject_dir: str):
    pattern = F"full_annotation/{subject_dir}/[!README]*.txt"
    full_pattern = os.path.join(BIGHAND_DATASET_DIR, pattern)
    annotation_files = glob.glob(full_pattern)
    return annotation_files


def create_all():
    subject_dirs_paths = [f for f in BIGHAND_DATASET_DIR.iterdir() if f.is_dir()]
    subject_dirs = [f.stem for f in subject_dirs_paths]
    for subject_dir, subject_dir_path in zip(subject_dirs, subject_dirs_paths):
        subject_annot_files = annotation_files_for_subject(subject_dir)

        for annot_file in subject_annot_files:
            filename = os.path.basename(annot_file)
            save_path = os.path.join(BIGHAND_DATASET_DIR, 'box_annotation', subject_dir, filename)
            create_for_file(annot_file, save_path)


if __name__ == '__main__':
    source_path = os.path.join(BIGHAND_DATASET_DIR,
                               'full_annotation/Subject_1/1 75_loc_shift_made_by_qi_20180112_v2.txt')
    save_path = os.path.join(BIGHAND_DATASET_DIR, 'box_annotation/Subject_1/1 75.txt')

    create_for_file(source_path, save_path)
