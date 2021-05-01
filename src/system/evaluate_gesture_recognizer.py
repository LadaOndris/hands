import numpy as np
from sklearn.metrics import det_curve, precision_recall_curve

import src.utils.plots as plots
from src.datasets.custom.dataset import CustomDataset
from src.system.gesture_recognizer import GestureRecognizer
from src.utils.paths import CUSTOM_DATASET_DIR, DOCS_DIR, OTHER_DIR


def save_produced_metrics_on_custom_dataset(file_name):
    ds = CustomDataset(CUSTOM_DATASET_DIR, batch_size=8, left_hand_only=True)
    live_acceptance = GestureRecognizer(error_thresh=200, orientation_thresh=50, database_subdir='test3')
    jres, angles, pred_labels, true_labels = live_acceptance.produce_jres(ds)
    custom_dataset_jre_path = OTHER_DIR.joinpath(file_name)
    np.savez(custom_dataset_jre_path, jres=jres, angles=angles, pred_labels=pred_labels, true_labels=true_labels)


def evaluation_plots_from_saved_metrics(file_name):
    data = load_saved_metrics(file_name)
    jres = data['jres']
    true_labels = data['true_labels']

    true_labels[true_labels != '1'] = '0'

    fig_path = DOCS_DIR.joinpath('figures/evaluation/false_positives_and_negatives.png')
    fpr, fnr, thresholds = det_curve_threshold_based(true_labels, jres, pos_label='1')
    plots.plot_scores(thresholds, y=[fpr, fnr], labels=['False positive rate', 'False negative rate'],
                      fig_location=None)
    plots.roc_curve(fnr, fpr)

    fig_path = DOCS_DIR.joinpath('figures/evaluation/precision_recall_threshold.png')
    precision, recall, thresholds = precision_recall_curve_threshold_based(
        true_labels, jres, pos_label='1')
    plots.plot_scores(thresholds, y=[precision, recall], labels=['Precision', 'Recall'],
                      fig_location=None)


def jre_histogram_from_saved_metrics(file_name):
    data = load_saved_metrics(file_name)
    jres = data['jres']
    orient_diffs = data['angles']
    true_labels = data['true_labels']
    true_jres = jres[true_labels == '1']
    true_diffs = orient_diffs[true_labels == '1']

    fig_gesture1 = DOCS_DIR.joinpath('figures/evaluation/jre_histogram_gesture1.png')
    fig_nongesture = DOCS_DIR.joinpath('figures/evaluation/jre_histogram_nongesture.png')
    plots.histogram(true_jres, label='Joint Relation Error (Gesture 1)', range=(0, 500),
                    fig_location=None)
    fig_orientation_gesture1 = DOCS_DIR.joinpath('figures/evaluation/orientation_histogram_gesture1_right_hand.png')
    plots.histogram(true_diffs, label='Orientation difference [degrees]', range=(0, 90),
                    fig_location=fig_orientation_gesture1)

    none_gesture_jres = jres[true_labels == '0']
    none_gesture_diffs = orient_diffs[true_labels == '0']
    plots.histogram(none_gesture_jres, label='Joint Relation Error (No gesture)', range=(0, 500),
                    fig_location=None)
    fig_orientation_nongesture = DOCS_DIR.joinpath('figures/evaluation/orientation_histogram_nongesture_right_hand.png')
    plots.histogram(none_gesture_diffs, label='Orientation difference [degrees]', range=(0, 90),
                    fig_location=fig_orientation_nongesture)


def load_saved_metrics(file_name):
    custom_dataset_jre_path = OTHER_DIR.joinpath(file_name)
    data = np.load(custom_dataset_jre_path, allow_pickle=True)
    return data


def det_curve_threshold_based(y_true, y_pred, pos_label=1):
    true = y_true.ravel()
    pred = y_pred.ravel()
    pred_max = np.max(pred)
    pred_inversed = pred_max - pred
    fpr, fnr, thresholds = det_curve(true, pred_inversed, pos_label='1')
    thresholds = pred_max - thresholds
    return fpr, fnr, thresholds


def precision_recall_curve_threshold_based(y_true, y_pred, pos_label=1):
    """
    Computes precision and recall for all possible thresholds.
    The y_pred value is expected not to be a probability score.
    """
    # true_labels = np.array([1,1, 1, 0, 0, 0])
    # jres = np.array([80, 124, 148, 150, 156, 220])
    true = y_true.ravel()
    pred = y_pred.ravel()
    pred_max = np.max(pred)
    pred_inversed = pred_max - pred
    precision, recall, thresholds = precision_recall_curve(true, pred_inversed, pos_label=pos_label)
    thresholds = np.concatenate([pred_max - thresholds])
    return precision[:-1], recall[:-1], thresholds


if __name__ == '__main__':
    import time

    filename = 'custom_dataset_jres_right.npz'
    start = time.time()
    # evaluation_plots_from_saved_metrics(filename)
    jre_histogram_from_saved_metrics(filename)
    # save_produced_metrics_on_custom_dataset(filename)
    end = time.time()
    print("It took for X batches x 8 images:", end - start)
