import random
import tensorflow as tf

from wandb_utils.config import Config
from wandb.keras import WandbMetricsLogger


class TestModel:
    def __init__(self):
        self.config = Config(epoch=8, batch_size=256).config()
        self.config.learning_rate = 0.01
        # Define specific configuration below, they will be visible in the W&B interface
        # Start of config
        self.config.layer_1 = 512
        self.config.activation_1 = "relu"
        self.config.dropout = random.uniform(0.01, 0.80)
        self.config.layer_2 = 10
        self.config.activation_2 = "softmax"
        self.config.optimizer = "sgd"
        self.config.loss = "sparse_categorical_crossentropy"
        self.config.metrics = ["accuracy"]
        # End
        self.model = self.__build_model()
        self.__compile()
        self.__load_dataset()

    def __build_model(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(self.config.layer_1, activation=self.config.activation_1),
            tf.keras.layers.Dropout(self.config.dropout),
            tf.keras.layers.Dense(self.config.layer_2, activation=self.config.activation_2)
        ])
    
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
            callbacks=wandb_callbacks
        )

    def save(self, filepath):
        self.model.save(filepath)

