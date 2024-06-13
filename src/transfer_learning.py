import argparse
import os
from pathlib import Path

from models.efficentnetv2b0 import EfficientNetV2B0Model
from models.mobilenetv2 import MobilenetV2Model
from models.resnet50v2 import Resnet50V2Model
from models.test_model import TestModel
from trainer.trainer import Trainer


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=[
                        'resnet50', 'efficientnet', 'mobilenet'], help='Choose the type of model')
    parser.add_argument('--size', type=str, help='Choose dataset size')
    parser.add_argument('--offline', required=False, action='store_true')
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--with-checkpoints',
                        required=False, action='store_true')
    return parser.parse_args()


def load_model(args):
    input_shape = (int(args.size), int(args.size), 3)
    if args.model == 'resnet50':
        model = Resnet50V2Model(input_shape)
    elif args.model == 'efficientnet':
        model = EfficientNetV2B0Model(input_shape)
    elif args.model == 'mobilenet':
        model = MobilenetV2Model(input_shape)
    else:
        model = TestModel()
    return model


if __name__ == '__main__':
    args = load_args()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    model = load_model(args)

    trainer = Trainer(model,
                      Path(f'./data/resized_dataset_{args.size}_{args.size}/'),
                      job_name=f"{args.model}_TF")

    if args.epochs:
        trainer.overload_config({'epochs': args.epochs})

    trainer.fit(checkpoints_on_epochs=args.with_checkpoints)

    model.save(f"{args.model}_{args.size}_TF.keras")
