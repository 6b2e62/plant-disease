import argparse
from pathlib import Path

from model.efficentnetv2b0 import EfficientNetV2B0Model
from model.mobilenetv2 import MobilenetV2Model
from model.resnet50v2 import Resnet50V2Model
from model.test_model import TestModel

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=[
                        'resnet50', 'efficientnet', 'mobilenet'], help='Choose the type of model')
    parser.add_argument('--size', type=str, help='Choose dataset size')
    return parser.parse_args()

def load_model(args):
    size = args.size
    if args.model == 'resnet50':
        model = Resnet50V2Model(Path(f'./data/resized_dataset_{size}_{size}/'), job_name="ResNet50v2")
    elif args.model == 'efficientnet':
        model = EfficientNetV2B0Model(Path(f'./data/resized_dataset_{size}_{size}/'), job_name="EfficientNetV2B0")
    elif args.model == 'mobilenet':
        model = MobilenetV2Model(Path(f'./data/resized_dataset_{size}_{size}/'), job_name="MobileNetV2")
    else:
        model = TestModel()
    return model

if __name__ == '__main__':
    args = load_args()
    model = load_model(args)

    model.fit()
    model.save(f"{args.model}_{args.size}.keras")
