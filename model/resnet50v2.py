import tensorflow as tf

from .model import Model

class Resnet50V2Model(Model):
    job_config = {
        "batch_size": 64,
        "epoch": 15,
        "learning_rate": 0.01,
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
    }

    def _Model__add_classifier(self):
        layer = tf.keras.layers.GlobalAveragePooling2D()(self.model.layers[-1].output)
        dense = tf.keras.layers.Dense(38, activation='softmax')(layer)
        
        self.model.trainable = not self.transfer_learning

        model_with_classifier = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=dense
        )
        self.model = model_with_classifier

    def build_model(self):
        return tf.keras.applications.ResNet50V2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
        )
        