
# Gesture recognition system
Author: Ladislav Ondris

This project performs gesture recognition from depth images. 
It consists of hand detection, hand pose estimation, and gesture classification.


See demonstration videos, which are located in the `docs/` directory.

## Prerequisites

Python 3.7.10  

* Intel RealSense SR305 depth camera,
* or any color camera


## Installation

Install the required packages with:  
```
pip install -r requirements.txt
```

In case TensorFlow has a wrong dependency of gast, which may result in warning
or error messages, install 0.3.3 version of gast, which downgrades the package from version 0.4.0.

```
pip install gast==0.3.3
```

## Usage examples

The system requires that the user defines the gesture to be recognized, which
is described in Section *Preparation of gesture database*. For demonstration purposes,
the gesture database is already prepared for the gesture with an opened palm, 
fingers outstretched and apart.  

The usage of the real-time recognition from live images or from the custom dataset is shown in 
*Real-time gesture recognition*.

### Real-time gesture recognition

**For demonstration**, the directory named `color` is already present,
containing several representative gestures.

To start the gesture recognition system using gesture database stored in 
the `color` directory:  

```
python3 recognize.py color
```

The system plots figures similar to the following:  
<p float="left">
    <img src="./docs/readme/" alt="live_gesture1" width="220"/>
    <img src="./docs/readme/" alt="live_nongesture" width="220"/>
</p>


### Preparation of gesture database

Beware: the preparation of gesture database requires either a depth or color camera. 
You can **skip this section** because there is already a database 
called `color` and `depth` available. Both databases contain six gestures---numbers 0 through 5.

The database scanner uses predictor of hand keypoints and writes them
in a file. The captured poses then serve as representatives of that gesture.
The database is then used by `live_gesture_recognizer.py`.

To capture a gesture with label `1` into a `gesturesdb` directory with a scan
period of one second and SR305 camera:  
```
python3 database.py gesturesdb 1 10 --camera SR305
```

To capture a gesture with label `hi` into a `colordb` directory with a scan 
period of one second and color camera:
```
python3 database.py colordb hi 10
```

```
usage: scan_database.py [-h] [--scan-period SCAN_PERIOD] [--camera CAMERA]
                   [--hide-plot]
                   directory label count

positional arguments:
  directory             the name of the directory that should contain the
                        user-captured gesture database
  label                 the label of the gesture that is to be captured
  count                 the number of samples to scan

optional arguments:
  -h, --help            show this help message and exit
  --scan-period SCAN_PERIOD
                        intervals between each capture in seconds (default: 1)
  --camera CAMERA       the camera model in use for live capture (default:
                        SR305)
  --hide-plot           hide plots of the captured poses - not recommended
```

## System architecture

The system is designed from several components. Each of these components inherit from a base class,
which defines its interface. The abstract classes are defined in `system/components/base.py`.
This allows easy changes of implementations of components.

The components are:
* ImageSource
* CoordinatePredictor
  * Detector
  * Estimator
* GestureRecognizer
* Display

Implementations of these classes can be found in `system/components` in their corresponding folders.
The specific implementations are discussed below.

<span style="color:red;">how to change which component is used.</span>


### ImageSource implementations

#### Empty display
`EmptyDisplay` is an empty implementation of the abstract class. It can be used
if no results are wanted to be displayed.

#### Stdout display
`StdoutDisplay` prints the recognized gesture to standard output.

<span style="color:red;">Console screen.</span>

#### Opencv display
`OpencvDisplay` supports plotting the image, together with a label of the recognized gesture.
It can also display a rectangle as the result of hand detection or the specific keypoints.

<span style="color:red;">Image.</span>


### CoordinatePredictor solutions

Coordinate predictors, given an image, return a set of 3D coordinates, specifying the 
precise position of the hand in space. The repository contains two solutions—color-based and depth-based.

#### Color-based

Uses **Mediapipe** hands solution from Google (https://google.github.io/mediapipe/solutions/hands.html).
The implementation class is named `MediapipeCoordinatePredictor`.

#### Depth-based

Uses custom-trained networks—**Blazeface** hand detector, and **Blazepose** hand pose estimator.
The implementation is in the `TrackingCoordinatePredictor` class. Both models are written in
TensorFlow.

This solution could also be adapted for color-based tracking, but it might require a few
tweaks in the architecture or network configs. Also, new datasets would have to be acquired,
and the models would have to be trained from scratch on those datasets. That said, the Mediapipe solution
provides great results currently.


### GestureRecognizer implementations

Two ready-to-use gesture recognizers are located in `src/system/components/gesture_recognizers` directory.

#### RelativeDistanceGestureRecognizer
`RelativeDistanceGestureRecognizer` was 
created as the first gesture recognizer. It requires proper setting of thresholds.

#### MLPGestureRecognizer
`MLPGestureRecognizer` is a classifier that uses a MLP to classify gestures.
The MLPGestureRecognizer is the preferred way of gesture
recognition for higher accuracy. That said, from its nature it can't
provide any feedback on which fingers are wrongly placed.

#### Other gesture-recognition related code

Other gesture-recognition related code is located in the
`src/gestures` directory.

`visualization.py`
* Captured gesture database can be visualized using t-SNE or 
LDA.

`sklearn_classifiers.py`
* One can use any classifier from the sklearn library.
This script evaluates many classifiers from sklearn
library to determine,
which performs best on the given captured gestures.




## Project structure

### Top-level structure

    .
    ├── datasets                # Datasets (including gesture database)
    ├── docs                    # Demonstration videos, readme files, and images 
    ├── logs                    # Saved models' weights
    ├── text_src                # Latex source files of the thesis' text
    ├── src                     # Source files
    ├── LICENSE                 # MIT license
    ├── README.md               # Contents of this file
    ├── requirements.txt        # Package requirements 
    └── bachelors_thesis.pdf    # Text of the thesis

### Datasets

    datasets
    ├── bighand                     # Hand pose estimation dataset (not preprocessed)
    ├── cvpr15_MSRAHandGestureDB    # Hand pose estimatino dataset (is preprocessed)
    ├── handseg150k                 # Hand segmentation dataset (both hands)
    ├── simple_boxes                # Generated toy object detection dataset
    ├── custom                      # Created dataset for the evaluation of gesture recognition
    └── usecase                     # Contains gesture databases captured by the user 

### Source files

    src
    ├── acceptance               # Gesture acceptance module (gesture recognition algorithm)
    ├── datasets                 # Dataset related code (pipelines, plots, generation)
    ├── detection                # Detection methods - Tiny YOLOv3, RDF
    ├── estimation               # JGR-P2O estimation model and preprocessing
    ├── metacentrum              # Scripts for training models in Metacentrum
    ├── system                   # Access point to gesture recognition system 
    │                              (database_scanner, gesture_recognizer, hand_position_estimator)
    └── utils                    # Camera, logs, plots, live capture, config

## License

This project is licensed under the terms of the MIT license.
