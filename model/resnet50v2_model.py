import tensorflow as tf

import wandb
from wandb.keras import WandbMetricsLogger


class Resnet50Model:
    def __init__(self):
        config = {
            "epoch": 5,
            "learning_rate": 0.01,
            "optimizer": "sgd",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        }
        settings = wandb.Settings(job_name="resnet50v2-model-job")
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
        return tf.keras.applications.ResNet50V2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            # output - odblokować ostatnią warstwę freeze
            # zobaczyc czy dziala to by default, czy wewn. warstwy są frozen, i czy ost. jest dynamiczna
        )

    def __compile(self):
        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

    def __load_dataset(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (
            tf.keras.datasets.cifar10.load_data()
        )
        self.x_train = self.x_train.astype("float32") / 255.0
        self.x_test = self.x_test.astype("float32") / 255.0

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
            callbacks=wandb_callbacks,
        )

    def save(self, filepath):
        self.model.save(filepath)