import tensorflow as tf

from .model import Model
import numpy as np

class EfficientNetV2B0Model(Model):
    job_config = {
            "batch_size": 64,
            "epoch": 5,
            "learning_rate": 0.01,
            "optimizer": "sgd",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
    }

    def build_model(self):
        return tf.keras.applications.EfficientNetV2B0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )