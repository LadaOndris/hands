import tensorflow as tf

from src.datasets.bighand.dataset import BighandDataset
from src.estimation.blazepose.data.preprocessing import preprocess
from src.utils.camera import CameraBighand
from src.utils.paths import BIGHAND_DATASET_DIR
from src.utils.plots import plot_image_with_skeleton


def get_line(file):
    with open(file, 'r') as f:
        return f.readline()


def show_sample_from_each_folder(save_fig_location_pattern=None):
    ds = BighandDataset(BIGHAND_DATASET_DIR, batch_size=1, shuffle=False)
    camera = CameraBighand()
    samples_per_file = 1
    for i, file in enumerate(ds.train_annotation_files):
        fig_location = str(save_fig_location_pattern).format(i)
        with open(file, 'r') as f:
            for sample_id in range(samples_per_file):
                line = f.readline()
                image, joints = ds._prepare_sample(line)
                image = tf.squeeze(image)
                joints2d = camera.world_to_pixel(joints)
                plot_image_with_skeleton(image, joints2d, fig_location=fig_location)
        print(F"{i}:", file)


def show_image(annotation_filepath: str, image_name: str):
    camera = CameraBighand()
    ds = BighandDataset(BIGHAND_DATASET_DIR, batch_size=1, shuffle=False)
    annotation_paths = [annot_file for annot_file in ds.train_annotation_files if annotation_filepath in annot_file]
    if len(annotation_paths) != 1:
        raise ValueError(F"Expected only a single annotation file. Got {len(annotation_paths)}")

    with open(annotation_paths[0], 'r') as file:
        lines = file.readlines()
        image_lines = [line for line in lines if image_name in line]

    if len(image_lines) != 1:
        raise ValueError(F"Expected only a single image line. Got {len(image_lines)}")
    image_line = image_lines[0]
    image_line_tensor = tf.convert_to_tensor(image_line)
    image, joints = ds._prepare_sample(image_line_tensor)

    # joints2d = camera.world_to_pixel(joints)
    # plot_image_with_skeleton(tf.squeeze(image), joints2d)

    norm_image, (joints, heatmaps) = preprocess(image, joints, camera, joints_type='xyz', heatmap_sigma=2,
                                                cube_size=180)
    plot_image_with_skeleton(norm_image, joints * 256)

    # config_path = SRC_DIR.joinpath('estimation/blazepose/configs/config_blazepose_heatmap.json')
    # with open(config_path, 'r') as f:
    #     config = json.load(f)
    # model_config = config["model"]
    #
    # model = ModelCreator.create_model(model_config["model_type"],
    #                                   model_config["num_keypoints"],
    #                                   model_config['num_keypoint_features'])
    #
    # outputs = model(norm_image[tf.newaxis, ...])
    #
    # tf.print(tf.shape(outputs))
    pass


if __name__ == '__main__':
    # show_sample_from_each_folder()
    show_image('Subject_4/76 150', 'image_D00002019.png')
