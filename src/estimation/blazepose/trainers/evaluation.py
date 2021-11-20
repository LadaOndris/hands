import json

import tensorflow as tf

import src.utils.plots as plots
from estimation.blazepose.models.ModelCreator import ModelCreator
from src.datasets.bighand.dataset import BighandDataset
from src.estimation.blazepose.data.preprocessing import preprocess
from src.utils.camera import CameraBighand
from src.utils.paths import BIGHAND_DATASET_DIR, ROOT_DIR, SRC_DIR


def evaluate(config):
    model_config = config['model']
    train_config = config["train"]
    test_config = config["test"]
    model = ModelCreator.create_model(model_config["model_type"],
                                      model_config["num_keypoints"],
                                      model_config['num_keypoint_features'])
    print("Loading model weights: " + test_config["weights_path"])
    weights_path = ROOT_DIR.joinpath(test_config["weights_path"])
    model.load_weights(weights_path)

    print("Preparing dataset...")
    camera = CameraBighand()
    prepare_fn = lambda image, joints: preprocess(image, joints, camera, joints_type='xyz',
                                                  heatmap_sigma=model_config['heatmap_kp_sigma'],
                                                  cube_size=model_config['cube_size'],
                                                  generate_random_crop_prob=train_config['generate_random_crop_prob'],
                                                  random_angle_stddev=train_config['random_angle_stddev'],
                                                  shift_stddev=train_config['shift_stddev'])
    ds = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=1, shuffle=True)

    for batch in ds.test_dataset:
        image, joints = batch[0][0], batch[1][0]
        normalized_image, y_true = prepare_fn(image, joints)
        true_joints = y_true['joints'][..., :21]

        pred_joints_batch, pred_heatmap_batch, pred_presence_batch = model.predict(normalized_image[tf.newaxis, ...])
        pred_joints = pred_joints_batch[0]
        pred_heatmap = pred_heatmap_batch[0]
        # plots.plot_image_with_skeleton(normalized_image, true_joints[:, :2] * 256)
        plots.plot_image_with_skeleton(normalized_image, pred_joints[:, :2] * 256)
        pass


if __name__ == "__main__":
    config_path = SRC_DIR.joinpath('estimation/blazepose/configs/config_blazepose.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    evaluate(config)
