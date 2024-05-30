from pathlib import Path
from typing import Optional

import optuna
import tensorflow as tf
from optuna_integration.wandb import WeightsAndBiasesCallback
from tensorflow import keras
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

    def __init__(self, ds_path: Path,
                 job_name: str,
                 transfer_learning: bool = True):
        self.ds_path = ds_path
        self.transfer_learning = transfer_learning

        start_from_checkpoint = False,
        checkpoints_on_epochs = False

        self.wandb_settings = wandb.Settings(job_name=job_name)

        self.train_ds, self.valid_ds, self.test_ds = None, None, None
        self.input_shape = None

        self.__init_wandb()
        print(self.job_config)

        self.__load_dataset(ds_path, self.job_config.batch_size)
        self.model = self.build_model()
        self.__add_classifier()
        self.__compile()
        self.model.summary()

    def __init_wandb(self):
        wandb.init(
            project="Detection of plant diseases",
            entity="uczenie-maszynowe-projekt",
            config=self.job_config,
            settings=self.wandb_settings
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
        elif self.job_config.optimizer == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.job_config.learning_rate)
        elif self.job_config.optimizer == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=self.job_config.learning_rate)
        else:
            raise ValueError("Unknown optimizer")
        self.model.compile(
            optimizer=optimizer,
            loss=self.job_config.loss,
            metrics=self.job_config.metrics,
        )

    def __load_dataset(self, ds_path: Path, batch_size: int):
        '''
        Load the dataset.
        '''
        self.train_ds = Dataset(self.__class__,
                                ds_path / "train", batch_size)
        self.valid_ds = Dataset(self.__class__,
                                ds_path / "valid", batch_size)
        self.test_ds = Dataset(self.__class__,
                               ds_path / "test", batch_size)
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

    def overload_config(self, epoch=15):
        self.job_config.update({"epoch": epoch}, allow_val_change=True)

    def fit(self,
            save_best_model: bool = False,
            checkpoints_on_epochs: bool = False):
        wandb_callbacks = [
            WandbMetricsLogger(log_freq=5)
        ]

        if checkpoints_on_epochs:
            checkpoint_filepath = f'checkpoint_{self.__class__.__name__}.weights.h5'
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                save_freq='epoch'
            )
            wandb_callbacks.append(model_checkpoint_callback)
        if save_best_model:
            best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f'best_{self.__class__.__name__}.weights.h5',
                save_weights_only=True,
                monitor='val_accuracy',
                save_best_only=True
            )
            wandb_callbacks.append(best_checkpoint_callback)

        print("Job config before model fit:\n\t", self.job_config)
        return self.model.fit(
            self.train_ds.get_dataset(),
            validation_data=self.valid_ds.get_dataset(),
            epochs=self.job_config.epochs,
            batch_size=self.job_config.batch_size,
            callbacks=wandb_callbacks,
        )

    def __optuna_objective(self, trial: optuna.trial.Trial):
        '''Optuna objective to be minimalized.'''
        learning_rate = trial.suggest_float("learning rate",
                                            low=1e-6,
                                            high=0.05,
                                            step=1e-4)
        batch_size = trial.suggest_categorical(
            "batch size", [8, 16, 32, 48, 64, 96, 128])
        epochs = trial.suggest_int("epochs", 20, 30)
        optimizer = trial.suggest_categorical(
            "optimizer", ["adam", "sgd", "rmsprop", "adagrad"])

        self.__load_dataset(self.ds_path, batch_size)

        wandb.config.update({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer
        }, allow_val_change=True)
        self.job_config = wandb.config

        # clear clutter from previous Keras session graphs.
        keras.backend.clear_session()

        self.__compile()
        if self.start_from_checkpoint:
            self.load_weights(
                f'checkpoint_{self.__class__.__name__}.weights.h5')
        self.fit(save_best_model=True,
                 checkpoints_on_epochs=self.checkpoints_on_epochs)

        score = self.model.evaluate()
        wandb.log({
            "score": score[1]
        })

        return score[1]

    def optuna_train(self,
                     n_trials: int = 10,
                     start_from_checkpoint: bool = False,
                     checkpoints_on_epochs: bool = False):
        '''
        Train the model using Optuna.
            - `n_trials` - number of optuna cycles
        '''
        self.start_from_checkpoint = start_from_checkpoint
        self.checkpoints_on_epochs = checkpoints_on_epochs

        self.study = optuna.create_study(direction="maximize")
        self.score = 0
        optuna_objective = self.__optuna_objective

        optuna_callbacks = []
        wandbc = WeightsAndBiasesCallback(
            wandb_kwargs={
                "project": "Detection of plant diseases",
                "entity": "uczenie-maszynowe-projekt",
            },
            as_multirun=True
        )
        optuna_objective = wandbc.track_in_wandb()(optuna_objective)
        optuna_callbacks.append(wandbc)

        self.study.optimize(optuna_objective,
                            n_trials=n_trials,
                            callbacks=optuna_callbacks)

    def evaluate(self):
        return self.model.evaluate(self.test_ds.get_dataset())

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save(self, filepath):
        self.model.save(filepath)

    def predict(self, img):
        return self.model.predict(img)
