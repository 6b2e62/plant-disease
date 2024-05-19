import argparse
from pathlib import Path

from model.efficentnetv2b0 import EfficientNetV2B0Model
from model.mobilenetv2 import MobilenetV2Model
from model.resnet50v2 import Resnet50Model
from model.test_model import TestModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=[
                    'resnet50', 'efficientnet', 'mobilenet'], help='Choose the type of model')
args = parser.parse_args()

print("STARTING")

if __name__ == '__main__':
    if args.model == 'resnet50':
        model = Resnet50Model()
    elif args.model == 'efficientnet':
        model = EfficientNetV2B0Model()
    elif args.model == 'mobilenet':
        model = MobilenetV2Model(Path('./data/resized_dataset_96_96/'))
    else:
        model = TestModel()

    model.fit()
