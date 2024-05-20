import tensorflow as tf

from .model import Model

class EfficientNetV2B0Model(Model):
    job_config = {
            "epoch": 5,
            "learning_rate": 0.01,
            "optimizer": "sgd",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
    }

    def build_model(self):
        return tf.keras.applications.EfficientNetV2B0(
            nput_shape=(128, 128, 3),
            include_top=not self.transfer_learning,
            weights=self.weights,
            input_tensor=None,
            pooling=None,
            classifier_activation="softmax"
        )