from model.test_model import TestModel
from model.resnet_50_model import Resnet50Model
from pathlib import Path
from dataset.dataset import Dataset

model = TestModel()
history = model.fit()