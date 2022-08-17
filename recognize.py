import argparse

from src.system.components.base import Display, ImageSource
from src.system.components.gesture_recognizers.MLPGestureRecognizer import MLPGestureRecognizer
from src.system.components.gesture_recognizers.RelativeDistanceGestureRecognizer import \
    RelativeDistanceGestureRecognizer
from src.system.main import System
from src.utils.camera import get_camera


def get_depth_filters():
    import pyrealsense2 as rs
    decimation_filter = rs.decimation_filter()
    decimation_filter.set_option(rs.option.filter_magnitude, 1)

    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 2)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
    spatial_filter.set_option(rs.option.holes_fill, 1)

    postprocessing_filters = [
        spatial_filter
    ]
    return postprocessing_filters


def get_display(display_name: str) -> Display:
    if display_name == 'opencv':
        from src.system.components.displays.OpencvDisplay import OpencvDisplay
        return OpencvDisplay()
    if display_name == 'stdout':
        from src.system.components.displays.StdoutDisplay import StdoutDisplay
        return StdoutDisplay()
    if display_name == 'empty':
        from src.system.components.displays.EmptyDisplay import EmptyDisplay
        return EmptyDisplay()
    raise NameError(f'Invalid display name: {display_name}')


def get_image_source(source_name: str, video_file_path: str) -> ImageSource:
    if source_name == 'primary':
        from src.system.components.image_sources.DefaultVideoCaptureSource import DefaultVideoCaptureSource
        return DefaultVideoCaptureSource()
    if source_name == 'realsense':
        from src.system.components.image_sources.RealSenseCameraWrapper import RealSenseCameraWrapper
        realsense_wrapper = RealSenseCameraWrapper(enable_depth=True, enable_color=False,
                                                   filters=get_depth_filters())
        return realsense_wrapper.get_depth_image_source()
    if source_name == 'video':
        if video_file_path is None:
            raise ValueError("Video file path must be set")
        from src.system.components.image_sources.VideoImageSource import VideoImageSource
        return VideoImageSource(video_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, action='store',
                        help='the name of the directory containg the user-captured gesture database')
    parser.add_argument('--source', type=str, action='store', default='primary',
                        help='the image source to be used (available: primary, realsense, video. Default: primary)')
    parser.add_argument('--display', type=str, action='store', default='opencv',
                        help='the diplay to be used (available: opencv, stdout, empty. Default: opencv)')
    parser.add_argument('--video', type=str, action='store', default=None,
                        help='a video file path to be used as image source (default: None)')
    parser.add_argument('--camera', type=str, action='store', default=None,
                        help='the camera model in use for live capture (default: None)')
    parser.add_argument('--error-threshold', type=int, action='store', default=150,
                        help='the pose (JRE) threshold (default: 120)')
    parser.add_argument('--orientation-threshold', type=int, action='store', default=90,
                        help='the orientation threshold in angles (maximum: 90, default: 90)')
    args = parser.parse_args()

    if args.source == 'realsense':
        from src.system.components.coordinate_predictors.TrackingCoordinatePredictor import TrackingCoordinatePredictor
        from src.system.components.detector import BlazehandDetector
        from src.system.components.estimator import BlazeposeEstimator
        from src.system.components.keypoints_to_rectangle import KeypointsToRectangleImpl

        # Uses depth camera and custom-trained neural nets
        camera = get_camera(args.camera)
        predictor = TrackingCoordinatePredictor(detector=BlazehandDetector(),
                                                estimator=BlazeposeEstimator(camera, presence_threshold=0.3),
                                                keypoints_to_rectangle=KeypointsToRectangleImpl(shift_coeff=0.1),
                                                camera=camera)
    else:
        from src.system.components.coordinate_predictors.MediapipeCoordinatePredictor import \
            MediapipeCoordinatePredictor

        # Uses color camera and MediaPipe tracker
        predictor = MediapipeCoordinatePredictor()

    image_source = get_image_source(args.source, args.video)
    display = get_display(args.display)
    simple_recognizer = RelativeDistanceGestureRecognizer(args.error_threshold, args.orientation_threshold,
                                                          args.directory)
    regression_recognizer = MLPGestureRecognizer(args.directory)

    system = System(image_source=image_source,
                    predictor=predictor,
                    display=display,
                    gesture_recognizer=regression_recognizer)
    system.start()
