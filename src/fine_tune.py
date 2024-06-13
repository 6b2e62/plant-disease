
import tensorflow as tf
from pathlib import Path
import argparse

from src.models.efficentnetv2b0 import EfficientNetV2B0Model
from src.models.mobilenetv2 import MobilenetV2Model
from src.models.resnet50v2 import Resnet50V2Model
from src.models.test_model import TestModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=[
                    'resnet50', 'efficientnet', 'mobilenet'], help='Choose the type of model')
parser.add_argument('--epochs', type=int, required=False)
parser.add_argument('--with-checkpoints', required=False, action='store_true')
parser.add_argument('--data-path', required=str, required=False)
parser.add_argument('--model-path', required=str, required=False)
args = parser.parse_args()

size = args.size
if args.model == 'resnet50':
    model = Resnet50V2Model(Path(args.data_path), job_name="ResNet50v2")
elif args.model == 'efficientnet':
    model = EfficientNetV2B0Model(Path(args.data_path), job_name="EfficientNetV2B0")
elif args.model == 'mobilenet':
    model = MobilenetV2Model(Path(args.data_path), job_name="MobileNetV2")
else:
    model = TestModel()

model.load_weights(f"{args.model_path}.keras")

model_class = model.__class__
if model_class == model.mobilenetv2.MobilenetV2Model:
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
elif model_class == model.resnet50v2.Resnet50V2Model:
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
elif model_class == model.efficentnetv2b0.EfficientNetV2B0Model:
    preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
else:
    print("Model not recognized.")
    exit(1)

if args.epochs:
    model.overload_config({'epochs': args.epochs})
    
model.fit(checkpoint=args.with_checkpoints)
model.save(f"{args.model}_256.keras")