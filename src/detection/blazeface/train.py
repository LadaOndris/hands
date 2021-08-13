import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import src.utils.logs as logs_utils
from src.datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from src.datasets.simple_boxes.dataset_bboxes import SimpleBoxesDataset
from src.datasets.tvhand.dataset import TvhandDataset
from src.detection.blazeface.loss import BlazeFaceLoss
from src.detection.blazeface.model import build_blaze_face
from src.detection.blazeface.utils import train_utils
from src.utils import bbox_utils
from src.utils.paths import HANDSEG_DATASET_DIR, LOGS_DIR, SIMPLE_DATASET_DIR, TVHAND_DATASET_DIR


def get_dataset(name: str, hyper_params):
    prior_boxes = bbox_utils.generate_prior_boxes(hyper_params['feature_map_shapes'],
                                                  hyper_params['aspect_ratios'])
    total_bboxes = tf.shape(prior_boxes)[-2]
    output_shape = (tf.TensorShape([total_bboxes, 4]), tf.TensorShape([total_bboxes, 1]))
    name = name.lower()
    if name == 'tvhand':
        return TvhandDataset(
            TVHAND_DATASET_DIR,
            out_img_size=hyper_params['img_size'],
            batch_size=hyper_params['batch_size'],
            prepare_output_fn=train_utils.prepare_expected_output_fn(prior_boxes, hyper_params),
            prepare_output_shape=output_shape)
    if name == 'handseg':
        return HandsegDatasetBboxes(
            HANDSEG_DATASET_DIR,
            train_size=0.8,
            img_size=[hyper_params['img_size'], hyper_params['img_size']],
            batch_size=hyper_params['batch_size'],
            prepare_output_fn=train_utils.prepare_expected_output_fn(prior_boxes, hyper_params),
            prepare_output_shape=output_shape)
    if name == 'simple':
        return SimpleBoxesDataset(
            SIMPLE_DATASET_DIR,
            train_size=0.8,
            img_size=[hyper_params['img_size'], hyper_params['img_size']],
            batch_size=hyper_params['batch_size'],
            prepare_output_fn=train_utils.prepare_expected_output_fn(prior_boxes, hyper_params),
            prepare_output_shape=output_shape)


def train(dataset_name: str, weights_path: str = None, debug=False, **kwargs):
    hyper_params = train_utils.get_hyper_params(**kwargs)
    channels = 3 if dataset_name == 'tvhand' else 1
    model = build_blaze_face(hyper_params['detections_per_layer'], channels)
    print(model.summary(line_length=150))
    if weights_path is not None:
        model.load_weights(weights_path)

    dataset = get_dataset(dataset_name, hyper_params)

    monitor_loss = 'val_loss'
    if dataset.num_test_batches == 0:
        dataset.test_dataset = None
        monitor_loss = 'loss'
    dataset.num_train_batches = 400

    log_dir = logs_utils.make_log_dir()
    checkpoint_path = logs_utils.compose_ckpt_path(log_dir)
    callbacks = [
        TensorBoard(log_dir=log_dir),
        ModelCheckpoint(filepath=checkpoint_path, monitor=monitor_loss, save_weights_only=True),
        EarlyStopping(monitor=monitor_loss, patience=10, restore_best_weights=True),
        TerminateOnNaN()
    ]

    # Don't iterate over the whole BigHand as a single epoch.
    # That would be endless waiting.
    # steps_per_epoch = 4096
    lr_schedule = ExponentialDecay(
        initial_learning_rate=hyper_params['learning_rate'],
        decay_steps=dataset.num_train_batches,
        decay_rate=hyper_params['learning_decay_rate'],
        staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)
    loss = BlazeFaceLoss(neg_pos_ratio=3, loc_loss_alpha=1)
    model.compile(optimizer=optimizer, loss=[loss.box_loss, loss.conf_loss])

    model.fit(dataset.train_dataset, epochs=1000, verbose=int(debug), callbacks=callbacks,
              steps_per_epoch=dataset.num_train_batches,
              validation_data=dataset.test_dataset,
              validation_steps=dataset.num_test_batches)

    # probably won't come to this, but just to be sure.
    model_filepath = logs_utils.compose_model_path(prefix=F"blazeface_{dataset_name.lower()}_")
    model.save_weights(model_filepath)
    # checkpoints are located in the log_dir
    # the saved model is located in the model_filepath
    return log_dir, str(model_filepath)


if __name__ == "__main__":
    weights = LOGS_DIR.joinpath('20210813-153226/train_ckpts/weights.01.h5')
    weights = None
    train('handseg', debug=True, batch_size=4, weights_path=weights)
