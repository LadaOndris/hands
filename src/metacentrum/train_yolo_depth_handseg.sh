#!/bin/bash
#PBS -N YoloV3Depth-Handseg
#PBS -q gpu
#PBS -l select=1:ncpus=24:ngpus=1:mem=48gb:cpu_flag=avx512dq:scratch_local=25gb
#PBS -l walltime=20:00:00
#PBS -m abe

DATADIR=/storage/brno6/home/ladislav_ondris/IBT
SCRATCHDIR="$SCRATCHDIR/IBT"
mkdir $SCRATCHDIR

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add python-3.6.2-gcc
module add python36-modules-gcc
module add opencv-3.4.5-py36
module add tensorflow-2.0.0-gpu-python3

cp -r "$DATADIR/src" "$SCRATCHDIR/" || { echo >&2 "Couldnt copy srcdir to scratchdir."; exit 2; }

# Prepare directory
mkdir "$SCRATCHDIR/datasets"
# Copy cvpr15_MSRAHandGestureDB.tar
cp -r "$DATADIR/datasets/handseg150k.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy handseg150k.tar"; exit 2; }
# Extract it
tar -xf "$SCRATCHDIR/datasets/handseg150k.tar" -C "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt extract handseg150k.tar"; exit 2; }

export PYTHONPATH=$SCRATCHDIR
python3 $SCRATCHDIR/train_yolov3depth.py

cp -r $SCRATCHDIR/logs $DATADIR/ || { echo >&2 "Couldnt copy logs to datadir."; exit 3; }
cp -r $SCRATCHDIR/saved_models $DATADIR/ || { echo >&2 "Couldnt copy saved_models to datadir."; exit 3; }
clean_scratch
