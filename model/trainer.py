from pathlib import Path

import optuna
import tensorflow as tf
from optuna_integration.wandb import WeightsAndBiasesCallback
from tensorflow import keras
from wandb.integration.keras import WandbMetricsLogger

import wandb
from dataset.dataset import Dataset
from model.efficentnetv2b0 import EfficientNetV2B0Model
from model.mobilenetv2 import MobilenetV2Model
from model.resnet50v2 import Resnet50V2Model

from .model import Model


class Trainer:
    '''
    Trainer class for the model.

    Params:
    - `model: Model` - Model to be trained.
    - `ds_path: Path` - Path to the dataset.
    - `job_name: str` - Name of the job.
    - `transfer_learning: bool` - Whether to use transfer learning or not.
    - `with_wandb: bool` - Whether to use Weights and Biases or not.
    '''

    def __init__(self,
                 model: Model,
                 ds_path: Path,
                 job_name: str,
                 transfer_learning: bool = True,
                 with_wandb: bool = True) -> None:
        self.ds_path = ds_path
        self.transfer_learning = transfer_learning

        self.model = model
        self.job_config = self.model.job_config

        self.preprocess_fn = self.__choose_preprocess_fn(model)

        self.start_from_checkpoint = False
        self.checkpoints_on_epochs = False
        self.save_best_model = False

        self.with_wandb = with_wandb
        if self.with_wandb:
            self.wandb_settings = wandb.Settings(job_name=job_name)
            self.__init_wandb()

        self.__load_dataset(ds_path, self.job_config.batch_size)

    def overload_config(self, config: dict):
        '''
        Overload the wandb config with new values.
        '''
        wandb.config.update(config, allow_val_change=True)
        self.job_config = wandb.config

    def fit(self,
            start_from_checkpoint: bool = False,
            save_best_model: bool = False,
            checkpoints_on_epochs: bool = False):
        '''
        Fit the model.

        Params:
        - `start_from_checkpoint` - whether to start from the last checkpoint
        - `save_best_model` - whether to save the best model
        - `checkpoints_on_epochs` - whether to save checkpoints on each epoch
        '''

        wandb_callbacks = [
            WandbMetricsLogger(log_freq=5)
        ]

        self.start_from_checkpoint = start_from_checkpoint
        self.save_best_model = save_best_model
        self.checkpoints_on_epochs = checkpoints_on_epochs

        if self.checkpoints_on_epochs:
            checkpoint_filepath = f'checkpoint_{self.model.__class__.__name__}.keras'
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_accuracy',
                save_freq='epoch'
            )
            wandb_callbacks.append(model_checkpoint_callback)

        if self.save_best_model:
            best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f'best_{self.model.__class__.__name__}.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
            wandb_callbacks.append(best_checkpoint_callback)

        if self.start_from_checkpoint:
            self.model.load_weights(
                f'checkpoint_{self.model.__class__.__name__}.keras')

        print(f'Run job_config:\n\t{self.job_config}\n\n')

        return self.model.fit(
            self.train_ds.get_dataset(),
            validation_data=self.valid_ds.get_dataset(),
            epochs=self.job_config.epochs,
            batch_size=self.job_config.batch_size,
            callbacks=wandb_callbacks,
        )

    def optuna_train(self,
                     n_trials: int = 10,
                     start_from_checkpoint: bool = False,
                     checkpoints_on_epochs: bool = False) -> None:
        '''
        Train the model using Optuna.

        Params:
            - `n_trials` - number of optuna cycles
            - `start_from_checkpoint` - whether to start from the last checkpoint
            - `checkpoints_on_epochs` - whether to save checkpoints on each epoch
        '''
        self.start_from_checkpoint = start_from_checkpoint
        self.checkpoints_on_epochs = checkpoints_on_epochs

        self.study = optuna.create_study(direction="maximize")
        self.score = 0
        optuna_objective = self.__optuna_objective
        self.model.save('initial_weights.keras')

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

    def __optuna_objective(self, trial: optuna.trial.Trial):
        '''Optuna objective to be minimalized.'''
        learning_rate = trial.suggest_float("learning rate",
                                            low=1e-6,
                                            high=0.02,
                                            step=1e-4)
        batch_size = trial.suggest_categorical(
            "batch size", [8, 16, 32, 48, 64, 96, 128])
        epochs = trial.suggest_int("epochs", 15, 30)
        optimizer = trial.suggest_categorical(
            "optimizer", ["adam", "sgd", "rmsprop", "adagrad"])

        self.__load_dataset(self.ds_path, batch_size)

        self.overload_config({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer
        })

        # clear clutter from previous Keras session graphs.
        keras.backend.clear_session()

        self.model.compile()

        self.model.load_weights('initial_weights.keras')

        self.fit(start_from_checkpoint=False,
                 save_best_model=True,
                 checkpoints_on_epochs=self.checkpoints_on_epochs)

        score = self.model.evaluate(self.test_ds.get_dataset())
        wandb.log({
            "score": score[1]
        })

        if score[1] > self.score:
            self.model.save(
                f'best_score_{self.model.__class__.__name__}.keras')
            self.score = score[1]
        return score[1]

    def __init_wandb(self):
        wandb.init(
            project="Detection of plant diseases",
            entity="uczenie-maszynowe-projekt",
            config=self.job_config,
            settings=self.wandb_settings
        )
        self.job_config = wandb.config

    def __load_dataset(self, ds_path: Path, batch_size: int):
        '''
        Load the dataset.
        '''
        self.train_ds = Dataset(
            ds_path / "train", self.preprocess_fn, batch_size)
        self.valid_ds = Dataset(
            ds_path / "valid", self.preprocess_fn, batch_size)
        self.test_ds = Dataset(
            ds_path / "test", self.preprocess_fn, batch_size)
        self.input_shape = self.train_ds.take(
            1).as_numpy_iterator().next()[0].shape[1:]

    @staticmethod
    def __choose_preprocess_fn(model: Model) -> callable:
        '''
        Choose preprocess function based on the model.
        '''
        model_class = model.__class__
        if model_class == MobilenetV2Model:
            return tf.keras.applications.mobilenet_v2.preprocess_input
        elif model_class == Resnet50V2Model:
            return tf.keras.applications.resnet_v2.preprocess_input
        elif model_class == EfficientNetV2B0Model:
            return tf.keras.applications.efficientnet_v2.preprocess_input
        else:
            return lambda x: x / 255.
