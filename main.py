import argparse

from file_manager.data_manager import DataManager
from model.efficentnetv2b0 import EfficientNetV2B0Model
from model.mobilenetv2 import MobilenetV2Model
from model.resnet50v2 import Resnet50Model
from model.test_model import TestModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=[
                    'resnet50', 'efficientnet', 'mobilenet'], help='Choose the type of model')
args = parser.parse_args()

if __name__ == '__main__':
    data_manager = DataManager()
    data_manager.download_data()
    data_manager.unzip_data("archive.zip", "original_dataset")
    data_manager.resize_dataset(shape=(96, 96), source="original_dataset")

    if args.model == 'resnet50':
        model = Resnet50Model()
    elif args.model == 'efficientnet':
        model = EfficientNetV2B0Model()
    elif args.model == 'mobilenet':
        model = MobilenetV2Model('./data/resized_dataset_96_96/')
    else:
        model = TestModel()

    model.fit()
