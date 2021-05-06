
# Gesture recognition system
Author: Ladislav Ondris

This project performs gesture recognition from depth images. 
It consists of hand detection, hand pose estimation, and gesture classification.
Hands are detected using a Tiny YOLOv3 model.
The gesture recognition system then uses a JGR-P2O hand pose estimator
to determine the hands' skeleton, which is used for gesture classification.


## Prerequisites

Python 3.7.10

## Installation

Install the required packages with:  
`pip install -r requirements.txt`

Install gast==0.3.3, which downgrades the package from version 0.4.0.
TensorFlow has a wrong dependency. It may otherwise print warnings and not function properly.  
`pip install gast==0.3.3`

## Usage examples

### System's usage

The system requires that the user defines the gesture to be recognized, which
is performed in *Preparation of gesture database*. The real-time recognition 
from live images or from the custom dataset is demonstrated in 
*Real-time gesture recognition*.

#### Preparation of gesture database

To capture a gesture with label `1` into a `gestures` directory with a scan period of one second and SR305 camera:  
`src/system/database_scanner.py --dir gestures --label 1 --scan-period 1 --camera SR305`

#### Real-time gesture recognition

To start the gesture recognition system using gesture database stored in the `gestures` directory:  
`src/system/gesture_recognizer.py --live --dir gestures --error-threshold 120 --orientation-threshold 60 --camera SR305`

To start the gesture recognition from the custom dataset:  
`src/system/gesture_recognizer.py --custom_dataset --dir gestures --error-threshold 120 --orientation-threshold 60 --camera SR305`

**For demonstration**, a directory named "test" is already present,
containing definitions for a gesture with opened palm with fingers outstretched
and apart.
`src/system/gesture_recognizer.py --live --dir test --error-threshold 120 --orientation-threshold 60 --camera SR305`

The system plots figures similar to the following:


### Hand detection




### Hand pose estimation



### Training of models

To train the JGR-P2O on the Bighand dataset:  
`src/estimation/train.py --train bighand`

To train the JGR-P2O model on the Bighand dataset from existing weights:  
`src/estimation/train.py --train bighand --model logs/20210426-125059/train_ckpts/weights.25.h5`

The number of features is set to 196 by default. Set the `--features` flag if required otherwise.

### Evaluation of models

To evaluate the trained JGR-P2O model on the MSRA dataset:  
`src/estimation/evaluation.py --dataset msra --model logs/20210421-221853/train_ckpts/weights.22.h5 --features 128`


## Project structure


## License

This project is licensed under the terms of the MIT license.