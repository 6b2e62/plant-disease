import argparse
import os
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
    parser.add_argument('--offline', required=False, action='store_true')
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--with-checkpoints', required=False, action='store_true')
    return parser.parse_args()

def load_model(args):
    size = args.size
    if args.model == 'resnet50':
        model = Resnet50V2Model(Path(f'./data/resized_dataset_{size}_{size}/'), job_name="ResNet50v2")
    elif args.model == 'efficientnet':
        model = EfficientNetV2B0Model(Path(f'./data/resized_dataset_{size}_{size}/'), job_name="EfficientNetV2B0")
    elif args.model == 'mobilenet':
        model = MobilenetV2Model(Path(f'./data/resized_dataset_{size}_{size}/'), transfer_learning=False, job_name="MobileNetV2_Beg")
    else:
        model = TestModel()
    return model

if __name__ == '__main__':
    args = load_args()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    model = load_model(args)

    if args.epochs:
        model.overload_config(epoch=args.epochs)

    model.optuna_train(checkpoints_on_epochs=args.with_checkpoints)
    model.save(f"{args.model}_{args.size}.keras")
