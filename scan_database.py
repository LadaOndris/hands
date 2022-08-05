import argparse

from src.system.components.CoordinatePredictor import MediapipeCoordinatePredictor, TrackingCoordinatePredictor
from src.system.components.detector import BlazehandDetector
from src.system.components.display import OpencvDisplay
from src.system.components.estimator import BlazeposeEstimator
from src.system.components.image_source import DefaultVideoCaptureSource, RealSenseCameraWrapper
from src.system.components.keypoints_to_rectangle import KeypointsToRectangleImpl
from src.system.database.scanner import UsecaseDatabaseScanner
from src.utils.camera import get_camera


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, action='store',
                        help='the name of the directory that should contain the user-captured gesture database')
    parser.add_argument('label', type=str, action='store',
                        help='the label of the gesture that is to be captured')
    parser.add_argument('count', type=int, action='store',
                        help='the number of samples to scan')

    parser.add_argument('--scan-period', type=float, action='store', default=1.0,
                        help='intervals between each capture in seconds (default: 1)')
    parser.add_argument('--camera', type=str, action='store', default=None,
                        help='the camera model in use for live capture (default: None -> VideoCapture(0) is selected)')
    parser.add_argument('--hide-plot', action='store_true', default=False,
                        help='hide plots of the captured poses - not recommended')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plot = not args.hide_plot

    display = OpencvDisplay()

    if args.camera is None:
        # Uses color camera and MediaPipe tracker
        image_source = DefaultVideoCaptureSource()
        predictor = MediapipeCoordinatePredictor()
    else:
        # Uses depth camera and custom-trained neural nets
        realsense_wrapper = RealSenseCameraWrapper(enable_depth=True, enable_color=False)
        image_source = realsense_wrapper.get_depth_image_source()
        camera = get_camera(args.camera)
        predictor = TrackingCoordinatePredictor(detector=BlazehandDetector(),
                                                estimator=BlazeposeEstimator(camera, presence_threshold=0.3),
                                                keypoints_to_rectangle=KeypointsToRectangleImpl(shift_coeff=0.1),
                                                camera=camera)

    scanner = UsecaseDatabaseScanner(subdir=args.directory,
                                     image_source=image_source,
                                     predictor=predictor,
                                     keypoints_to_rectangle=KeypointsToRectangleImpl(shift_coeff=0.1),
                                     display=display)
    scanner.scan_into_subdir(args.label, args.count, scan_period=args.scan_period)
