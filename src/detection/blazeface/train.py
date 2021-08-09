from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import src.utils.logs as logs_utils
from src.datasets.bighand.dataset import BighandDataset
from src.detection.blazeface.loss import BlazeFaceLoss
from src.detection.blazeface.model import build_blaze_face
from src.utils.paths import BIGHAND_DATASET_DIR


def train(batch_size: int, learning_rate: float, learning_decay_rate: float, weights_path: str = None):
    model = build_blaze_face(detections_per_layer=[6, 2, 2, 2], channels=1)
    print(model.summary(line_length=150))
    if weights_path is not None:
        model.load_weights(weights_path)

    dataset = BighandDataset(BIGHAND_DATASET_DIR, batch_size=batch_size, shuffle=True)
    monitor_loss = 'val_loss'
    if dataset.num_test_batches == 0:
        dataset.test_dataset = None
        monitor_loss = 'loss'

    log_dir = logs_utils.make_log_dir()
    checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
    callbacks = [
        TensorBoard(log_dir=log_dir, update_freq='epoch'),
        ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_loss, save_weights_only=True),
        EarlyStopping(monitor=monitor_loss, patience=10, restore_best_weights=True),
        TerminateOnNaN()
    ]

    # Don't iterate over the whole BigHand as a single epoch.
    # That would be endless waiting.
    steps_per_epoch = 4096
    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_decay_rate,
        staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)
    loss = BlazeFaceLoss()
    model.compile(optimizer=optimizer, loss=[loss.box_loss, loss.conf_loss])

    model.fit(dataset.train_dataset, epochs=1000, verbose=1, callbacks=callbacks,
              steps_per_epoch=steps_per_epoch,
              validation_data=dataset.test_dataset,
              validation_steps=dataset.num_test_batches)

    # probably won't come to this, but just to be sure.
    # (the best checkpoint is being saved after each epoch)
    model_filepath = logs_utils.compose_model_path(prefix=F"blazeface_bighand_")
    model.save_weights(model_filepath)
    # checkpoints are located in the log_dir
    # the saved model is located in the model_filepath
    return log_dir, str(model_filepath)
