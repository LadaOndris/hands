import argparse
import json

from src.estimation.blazepose.trainers.trainer import train
from src.utils.paths import SRC_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, action='store', default=None,
                    help='a config file name')
parser.add_argument('--verbose', type=int, action='store', default=1,
                    help='verbose training output')
parser.add_argument('--batch-size', type=int, action='store', default=64,
                    help='the number of samples in a batch')
args = parser.parse_args()


config_path = SRC_DIR.joinpath('estimation/blazepose/configs/', args.config)
with open(config_path, 'r') as f:
    config = json.load(f)
train(config, batch_size=args.batch_size, verbose=args.verbose)
