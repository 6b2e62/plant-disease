from model.test_model import TestModel
from model.resnet_50_model import Resnet50Model
from pathlib import Path
from dataset.dataset import Dataset

if __name__ == "__main__":
    # Loading dataset
    # train_dataset = Dataset(Path('data/resized_dataset/train'))
    # valid_dataset = Dataset(Path('data/resized_dataset/valid'))
    # for i in train_dataset.take(1):
    #     print(i)

    # Training model
    # model = TestModel()
    # history = model.fit()
    # model.save("./trained_models/test_model.keras")
