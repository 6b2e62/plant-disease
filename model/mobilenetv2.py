import tensorflow as tf

from .model import Model


class MobilenetV2Model(Model):
    job_config = {
        "epochs": 25,
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
    }

    def build_model(self):
        return tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights=self.weights,
        )
