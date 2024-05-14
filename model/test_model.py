import random
import tensorflow as tf

import wandb
from wandb.keras import WandbMetricsLogger


class TestModel:
    def __init__(self):
        config = {
            "epoch": 5,
            "batch_size": 256,
            "learning_rate": 0.01,
            "layer_1": 512,
            "activation_1": "relu",
            "dropout": 0.5,
            "layer_2": 10,
            "activation_2": "softmax",
            "optimizer": "sgd",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        }
        settings = wandb.Settings(job_name="test-model-job")
        wandb.init(
            project="Detection of plant diseases",
            entity="uczenie-maszynowe-projekt",
            config=config,
            settings=settings,
        )
        self.config = wandb.config
        self.model = self.__build_model()
        self.__compile()
        self.__load_dataset()

    def __build_model(self):
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(
                    self.config.layer_1, activation=self.config.activation_1
                ),
                tf.keras.layers.Dropout(self.config.dropout),
                tf.keras.layers.Dense(
                    self.config.layer_2, activation=self.config.activation_2
                ),
            ]
        )

    def __compile(self):
        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

    def __load_dataset(self):
        mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        self.x_train, self.y_train = self.x_train[::5], self.y_train[::5]
        self.x_test, self.y_test = self.x_test[::20], self.y_test[::20]

    def fit(self):
        wandb_callbacks = [
            WandbMetricsLogger(log_freq=5),
            # Not supported with Keras >= 3.0.0
            # WandbModelCheckpoint(filepath="models"),
        ]
        return self.model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=self.config.epoch,
            batch_size=self.config.batch_size,
            validation_data=(self.x_test, self.y_test),
            callbacks=wandb_callbacks,
        )

    def save(self, filepath):
        self.model.save(filepath)
