from pathlib import Path

RELATIVE_BASE_PATH = "../../"
_ROOT_PATH = Path(__file__).parent.parent.parent

ROOT_DIR = _ROOT_PATH
LOGS_DIR = _ROOT_PATH.joinpath('logs')
SAVED_MODELS_DIR = _ROOT_PATH.joinpath('saved_models')
MODELS_DIR = _ROOT_PATH.joinpath('models')
SRC_DIR = _ROOT_PATH.joinpath('src')
DOCS_DIR = _ROOT_PATH.joinpath('docs')
OTHER_DIR = _ROOT_PATH.joinpath('other')

HANDSEG_DATASET_DIR = _ROOT_PATH.joinpath("datasets/handseg150k")
MSRAHANDGESTURE_DATASET_DIR = _ROOT_PATH.joinpath("datasets/cvpr15_MSRAHandGestureDB")
BIGHAND_DATASET_DIR = _ROOT_PATH.joinpath("datasets/bighand")
CUSTOM_DATASET_DIR = _ROOT_PATH.joinpath("datasets/custom")
USECASE_DATASET_DIR = _ROOT_PATH.joinpath("datasets/usecase")
SIMPLE_DATASET_DIR = _ROOT_PATH.joinpath("datasets/simple_boxes")
MHP_DATASET_DIR = _ROOT_PATH.joinpath("datasets/mhp")
TVHAND_DATASET_DIR = _ROOT_PATH.joinpath("datasets/tvhand")
NYU_DATASET_DIR = _ROOT_PATH.joinpath("datasets/nyu")

USECASE_DATASET_JOINTS_PATH = USECASE_DATASET_DIR.joinpath('joints.txt')
YOLO_CONFIG_FILE = SRC_DIR.joinpath("detection/yolov3/architecture/yolov3-tiny.cfg")
YOLO_DEPTH_CONFIG_FILE = SRC_DIR.joinpath("detection/yolov3/architecture/yolov3-tiny-depth.cfg")
