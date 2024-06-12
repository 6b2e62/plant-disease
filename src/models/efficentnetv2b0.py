import tensorflow as tf

from .model import Model

class EfficientNetV2B0Model(Model):
    job_config = {
            "epochs": 25,
            "learning_rate": 0.005,
            "batch_size": 64,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
    }

    def build_model(self):
        return tf.keras.applications.EfficientNetV2B0(
            input_shape=self.input_shape,
            include_top=False,
            weights=self.weights,
        )