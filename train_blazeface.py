import argparse

from src.detection.blazeface.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, action='store',
                        help='the dataset to be used for training (allowed options: handseg, tvhand)')
    args = parser.parse_args()
    train(args.dataset)
