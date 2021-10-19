import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

import src.utils.logs as logs_utils
from src.datasets.bighand.dataset import BighandDataset
from src.estimation.blazepose.data.preprocessing import preprocess
from src.estimation.blazepose.metrics.mae import MeanAverageErrorMetric
from src.estimation.blazepose.metrics.mje import MeanJointErrorMetric
from src.estimation.blazepose.models.ModelCreator import ModelCreator
from src.estimation.blazepose.trainers.losses import euclidean_distance_loss, focal_loss, focal_tversky, get_huber_loss, \
    get_wing_loss, JointFeaturesLoss
from src.estimation.blazepose.trainers.TrainPhase import TrainPhase
from src.utils.camera import CameraBighand
from src.utils.paths import BIGHAND_DATASET_DIR, SRC_DIR


def get_loss_functions(config_losses):
    weights = config_losses['weights']
    heatmap_loss = get_loss_by_name(config_losses['heatmap_loss'])

    keypoint_losses_weights = config_losses['keypoint_losses']['weights']
    keypoint_coord_loss = get_loss_by_name(config_losses['keypoint_losses']['coords'])
    keypoint_presence_loss = get_loss_by_name(config_losses['keypoint_losses']['presence'])
    keypoint_loss = JointFeaturesLoss(keypoint_coord_loss, keypoint_presence_loss, keypoint_losses_weights)

    losses = {'heatmap': heatmap_loss,
              'joints': keypoint_loss}
    return losses, weights


def get_loss_by_name(loss_name: str):
    if loss_name == "euclidean_distance_loss":
        return euclidean_distance_loss
    elif loss_name == "focal_tversky":
        return focal_tversky
    elif loss_name == "huber":
        return get_huber_loss(delta=1.0, weights=(1.0, 1.0))
    elif loss_name == "focal":
        return focal_loss(gamma=2, alpha=0.25)
    elif loss_name == "wing_loss":
        return get_wing_loss()
    elif loss_name == 'binary_crossentropy':
        return tf.keras.losses.binary_crossentropy
    raise ValueError(F"Invalid loss name: '{loss_name}'")


def train(config, batch_size, verbose):
    """Train model
    Args:
        config (dict): Training configuration from configuration file
    """
    train_config = config["train"]
    test_config = config["test"]
    model_config = config["model"]

    model = ModelCreator.create_model(model_config["model_type"],
                                      model_config["num_keypoints"],
                                      model_config['num_keypoint_features'])

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

    losses, weights = get_loss_functions(train_config["losses"])

    camera = CameraBighand()
    hm_mae_metric = MeanAverageErrorMetric(name="mae1")
    kp_mae_metric = MeanAverageErrorMetric(name="mae2")
    kp_mje_metric = MeanJointErrorMetric(camera)
    model.compile(optimizer=Adam(train_config["learning_rate"]),
                  loss=losses, loss_weights=weights,
                  metrics={"heatmap": [hm_mae_metric], "joints": [kp_mae_metric, kp_mje_metric]})

    monitor_loss = 'val_loss'
    log_dir = logs_utils.make_log_dir()
    checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
    callbacks = [
        TensorBoard(log_dir=log_dir, update_freq='batch'),
        ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_loss, save_weights_only=True),
        EarlyStopping(monitor=monitor_loss, patience=10, restore_best_weights=True),
        TerminateOnNaN()
    ]

    prepare_fn = lambda image, joints: preprocess(image, joints, camera, joints_type='xyz', heatmap_sigma=2,
                                                  cube_size=180,
                                                  generate_random_crop_prob=train_config['generate_random_crop_prob'])
    prepare_fn_shape = (tf.TensorShape([256, 256, 1]), tf.TensorShape([21, 3]), tf.TensorShape([64, 64, 21]))
    ds = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=batch_size,
                        shuffle=True, prepare_output_fn=prepare_fn, prepare_output_fn_shape=prepare_fn_shape)

    model.fit(ds.train_dataset,
              epochs=train_config["nb_epochs"],
              steps_per_epoch=2000,
              validation_data=ds.test_dataset,
              validation_steps=ds.num_test_batches,
              callbacks=callbacks,
              verbose=verbose)


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


if __name__ == "__main__":
    config_path = SRC_DIR.joinpath('estimation/blazepose/configs/config_blazepose_heatmap.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    train(config, batch_size=8, verbose=1)
