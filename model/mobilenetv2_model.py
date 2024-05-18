import tensorflow as tf

from .model import Model


class MobilenetV2Model(Model):
    job_config = {
        "epoch": 5,
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy", "loss"],
    }

    def build_model(self):
        return tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=not self.transfer_learning,
            weights=self.weights,
            classifier_activation="softmax",

        )
