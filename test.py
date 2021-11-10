import json
import time
import tensorflow as tf
from src.datasets.bighand.dataset import BighandDataset
from src.estimation.blazepose.data.preprocessing import preprocess
from src.estimation.blazepose.models.ModelCreator import ModelCreator
from src.utils.camera import CameraBighand
from src.utils.paths import BIGHAND_DATASET_DIR, ROOT_DIR, SRC_DIR
from src.utils.plots import plot_depth_image, plot_image_with_skeleton

config_path = SRC_DIR.joinpath('estimation/blazepose/configs/config_blazepose_heatmap.json')
with open(config_path, 'r') as f:
    config = json.load(f)

test_config = config["test"]
model_config = config["model"]

model = ModelCreator.create_model(model_config["model_type"],
                                  model_config["num_keypoints"],
                                  model_config['num_keypoint_features'])
weights_path = ROOT_DIR.joinpath(test_config["pretrained_weights_path"])
model.load_weights(weights_path)

camera = CameraBighand()
prepare_fn = lambda image, joints: preprocess(image, joints, camera, joints_type='xyz', heatmap_sigma=2,
                                              cube_size=180, generate_random_crop_prob=0)

def run_embedded_preprocess():
    ds = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=1,
                        shuffle=True, prepare_output_fn=prepare_fn)

    for image, (joints, heatmaps) in ds.train_dataset:
        # plot_image_with_skeleton(image[0], joints[0] * 256)

        start_time = time.time()
        joints_pred, heatmaps_pred = model.predict(image)
        elapsed_time = time.time() - start_time
        print(F"Elapsed time: {elapsed_time:.3f}")

        plot_depth_image(heatmaps[0, :, :, 0])
        plot_depth_image(heatmaps_pred[0, :, :, 0])
        pass

def run_external_preprocess():
    ds = BighandDataset(BIGHAND_DATASET_DIR, test_subject="Subject_8", batch_size=1,
                        shuffle=True)
    for image, joints in ds.train_dataset:
        start_time = time.time()
        image_norm, (joints_norm, heatmaps) = prepare_fn(image[0], joints[0])
        elapsed_time = time.time() - start_time
        print(F"Elapsed preprocess time: {elapsed_time:.3f}")

        start_time = time.time()
        joints_pred, heatmaps_pred = model.predict(image_norm[tf.newaxis, ...])
        elapsed_time = time.time() - start_time
        print(F"Elapsed predict time: {elapsed_time:.3f}")

run_embedded_preprocess()