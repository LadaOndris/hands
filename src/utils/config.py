from src.utils.paths import SRC_DIR

YOLO_CONFIG_FILE = SRC_DIR.joinpath("core/cfg/yolov3-tiny.cfg")
YOLO_CONFIG_FILE_RECT = SRC_DIR.joinpath("core/cfg/yolov3-tiny-rect.cfg")
TRAIN_YOLO_IOU_IGNORE_THRES = .7
TRAIN_YOLO_CONF_THRESHOLD = .5
TEST_YOLO_CONF_THRESHOLD = .94

JGRJ2O_TRAIN_BATCH_SIZE = 32
JGRJ2O_TEST_BATCH_SIZE = 1
JGRJ2O_LEARNING_RATE = 0.0001
JGRJ2O_LR_DECAY = 0.96
JGRJ2O_WEIGHT_DECAY = 0.00005


