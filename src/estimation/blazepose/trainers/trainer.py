import os
import pathlib
import importlib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

from src.estimation.blazepose.models.ModelCreator import ModelCreator
from src.estimation.blazepose.trainers.losses import euclidean_distance_loss, focal_loss, focal_tversky, get_huber_loss, \
    get_wing_loss
from src.estimation.blazepose.trainers.TrainPhase import TrainPhase
import src.utils.logs as logs_utils
from src.estimation.blazepose.metrics.mae import get_mae_metric
from src.estimation.blazepose.metrics.pck import get_pck_metric


def train(config):
    """Train model
    Args:
        config (dict): Training configuration from configuration file
    """
    train_config = config["train"]
    test_config = config["test"]
    model_config = config["model"]

    model = ModelCreator.create_model(model_config["model_type"], model_config["num_keypoints"])

    if train_config["load_weights"]:
        print("Loading model weights: " +
              train_config["pretrained_weights_path"])
        model.load_weights(train_config["pretrained_weights_path"])

    train_phase = TrainPhase(train_config.get("train_phase", "UNKNOWN"))
    if train_phase == train_phase.HEATMAP:
        freeze_regression_layers(model)
    elif train_phase == train_phase.REGRESSION:
        freeze_heatmap_layers(model)

    print(model.summary())

    loss_functions = {
        "heatmap": train_config["heatmap_loss"],
        "joints": train_config["keypoint_loss"]
    }

    # Replace all names with functions for custom losses
    for k in loss_functions.keys():
        if loss_functions[k] == "euclidean_distance_loss":
            loss_functions[k] = euclidean_distance_loss
        elif loss_functions[k] == "focal_tversky":
            loss_functions[k] = focal_tversky
        elif loss_functions[k] == "huber":
            loss_functions[k] = get_huber_loss(delta=1.0, weights=(1.0, 1.0))
        elif loss_functions[k] == "focal":
            loss_functions[k] = focal_loss(gamma=2, alpha=0.25)
        elif loss_functions[k] == "wing_loss":
            loss_functions[k] = get_wing_loss()

    loss_weights = train_config["loss_weights"]

    hm_pck_metric = get_pck_metric(ref_point_pair=test_config["pck_ref_points_idxs"], thresh=test_config["pck_thresh"])(name="pck1")
    hm_mae_metric = get_mae_metric()(name="mae1")
    kp_pck_metric = get_pck_metric(ref_point_pair=test_config["pck_ref_points_idxs"], thresh=test_config["pck_thresh"])(name="pck2")
    kp_mae_metric = get_mae_metric()(name="mae2")
    model.compile(optimizer=Adam(train_config["learning_rate"], momentum=0.9),
                  loss=loss_functions, loss_weights=loss_weights,
                  metrics={"heatmap": [hm_pck_metric, hm_mae_metric], "joints": [kp_pck_metric, kp_mae_metric]})

    monitor_loss = 'val_loss'
    log_dir = logs_utils.make_log_dir()
    checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
    callbacks = [
        TensorBoard(log_dir=log_dir, update_freq='epoch'),
        ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_loss, save_weights_only=True),
        EarlyStopping(monitor=monitor_loss, patience=10, restore_best_weights=True),
        TerminateOnNaN()
    ]

    model.fit(train_dataset,
              epochs=train_config["nb_epochs"],
              steps_per_epoch=len(train_dataset),
              validation_data=val_dataset,
              validation_steps=len(val_dataset),
              callbacks=callbacks,
              verbose=1)


def freeze_regression_layers(model):
    print("Freezing regression layers:")
    for layer in model.layers:
        if layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False


def freeze_heatmap_layers(model):
    print("Freezing heatmap layers:")
    for layer in model.layers:
        if not layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False
