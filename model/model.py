from pathlib import Path
from typing import Optional

import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger

import wandb
from dataset.dataset import Dataset


class Model:
    '''
    Params:
    - `ds_path` - Path to the dataset.
    - `include_top` - Wheter to add classification layer.
                        Should be `False`, when doing transfer learning.
    - `weights` - Predefined weights to be used, either `None` for random
                    weights or `"imagenet"` for model trained on ImageNet.
    '''

    job_config = dict()

    def __init__(self, ds_path: Path, transfer_learning: bool = True, weights: Optional[str] = "imagenet"):
        self.transfer_learning = transfer_learning
        self.weights = weights
        self.wandb_settings = wandb.Settings(job_name="mobilenet")
        self.train_ds, self.valid_ds, self.test_ds = None, None, None
        self.input_shape = None

        self.__init_wandb()
        print(self.job_config)

        self.__load_dataset(ds_path)
        self.model = self.build_model()
        self.__add_classifier()
        self.__compile()
        self.model.summary()

    def __init_wandb(self):
        wandb.init(
            project="Detection of plant diseases",
            entity="uczenie-maszynowe-projekt",
            config=self.job_config,
            settings=self.wandb_settings,
        )
        self.job_config = wandb.config

    def build_model(self):
        '''
        To be implemented in the child class.
        '''
        return None

    def __compile(self):
        '''
        Compile the model.
        '''
        if self.job_config.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.job_config.learning_rate)
        elif self.job_config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.job_config.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=self.job_config.loss,
            metrics=self.job_config.metrics,
        )

    def __load_dataset(self, ds_path: Path):
        '''
        Load the dataset.
        '''
        self.train_ds = Dataset(
            ds_path / "train", batch_size=self.job_config.batch_size)
        self.valid_ds = Dataset(
            ds_path / "valid", batch_size=self.job_config.batch_size)
        self.test_ds = Dataset(
            ds_path / "test", batch_size=self.job_config.batch_size)
        self.input_shape = self.train_ds.take(
            1).as_numpy_iterator().next()[0].shape[1:]

    def __add_classifier(self):
        '''
        Add custom classifier to the model.
        '''
        average_pooling_2d = tf.keras.layers.GlobalAveragePooling2D()(
            self.model.layers[-1].output)
        dropout = tf.keras.layers.Dropout(0.2)(average_pooling_2d)
        dense = self.dense1 = tf.keras.layers.Dense(
            38, activation=tf.nn.softmax)(dropout)

        # Freeze the Model
        self.model.trainable = not self.transfer_learning

        model_with_classifier = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[dense])
        self.model = model_with_classifier

    def fit(self):
        wandb_callbacks = [
            WandbMetricsLogger(log_freq=5),
            # Not supported with Keras >= 3.0.0
            # WandbModelCheckpoint(filepath="models"),
        ]
        return self.model.fit(
            self.train_ds.get_dataset(),
            validation_data=self.valid_ds.get_dataset(),
            epochs=self.job_config.epoch,
            batch_size=self.job_config.batch_size,
            callbacks=wandb_callbacks,
        )

    def evaluate(self):
        return self.model.evaluate(self.test_ds.get_dataset())

    def save(self, filepath):
        self.model.save(filepath)
