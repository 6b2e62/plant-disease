import tensorflow as tf

from .model import Model


class MobilenetV2Model(Model):
    job_config = {
        "epoch": 25,
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "sgd",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
    }

    def build_model(self):
        return tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=not self.transfer_learning,
            weights=self.weights,
            classifier_activation="softmax",
        )
