import tensorflow as tf

from .model import Model
import numpy as np

class EfficientNetV2B0Model(Model):
    job_config = {
            "batch_size": 64,
            "epoch": 5,
            "learning_rate": 0.01,
            "optimizer": "sgd",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
    }

    def __add_classifier(self):
        self.model.trainable = False

        i = tf.keras.layers.Input([None, None, 3], dtype="uint8")
        x = np.ops.cast(i, "float32")
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        layer = tf.keras.layers.GlobalAveragePooling2D()(self.model(x))
        dense = tf.keras.layers.Dense(38, activation='softmax')(layer)
        
        model_with_classifier = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=dense
        )
        self.model = model_with_classifier

    def build_model(self):
        return tf.keras.applications.EfficientNetV2B0(
            input_shape=self.input_shape,
            include_top=False,
            weights=self.weights,
            classes=1000,
            classifier_activation='softmax',
            include_preprocessing=True
        )