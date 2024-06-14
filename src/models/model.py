from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from utils import dotdict


class Model:
    '''
    Base class model.

    Args:
        - `input_shape: (Tuple)`: Shape of the input image.
        - `transfer_learning: (bool)`: Whether to use transfer learning or not.
        - `double_classifier (bool)`: Whether to compile model with double output or not.
    '''

    job_config = dict()

    def __init__(self,
                 input_shape: Tuple,
                 transfer_learning: bool = True,
                 double_classifier: bool = False) -> None:
        self.transfer_learning = transfer_learning
        self.input_shape = input_shape
        self.job_config = dotdict(self.job_config)
        if double_classifier:
            self.job_config.loss = {'plant': 'categorical_crossentropy',
                                    'disease': 'categorical_crossentropy'}
            self.job_config.metrics = {'plant': 'accuracy',
                                       'disease': 'accuracy'}

        if self.transfer_learning:
            self.weights = 'imagenet'
        else:
            self.weights = None
        print("Weights: ", "random" if self.weights == None else self.weights)

        self.model = self.build_model()
        if double_classifier:
            self.__add_double_classifier()
        else:
            self.__add_classifier()
        self.compile()
        self.model.summary()

    def build_model(self):
        '''
        To be implemented in the child class.
        '''
        return None

    def compile(self):
        '''
        Compile the model.
        '''
        if self.job_config['optimizer'] == "sgd":
            optimizer = keras.optimizers.SGD(
                learning_rate=self.job_config.learning_rate)
        elif self.job_config['optimizer'] == "adam":
            optimizer = keras.optimizers.Adam(
                learning_rate=self.job_config.learning_rate)
        elif self.job_config['optimizer'] == "rmsprop":
            optimizer = keras.optimizers.RMSprop(
                learning_rate=self.job_config.learning_rate)
        elif self.job_config['optimizer'] == "adagrad":
            optimizer = keras.optimizers.Adagrad(
                learning_rate=self.job_config.learning_rate)
        else:
            raise ValueError("Unknown optimizer")
        self.model.compile(
            optimizer=optimizer,
            loss=self.job_config.loss,
            metrics=self.job_config.metrics,
        )

    def evaluate(self, X):
        return self.model.evaluate(X)

    def predict(self, X):
        return self.model.predict(X)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save(self, filepath):
        self.model.save(filepath)

    def __getattr__(self, attr):
        '''
        Get attribute from the keras model.
        It is used to get the model's methods e.g. fit.
        '''

        return getattr(self.model, attr)

    def __add_classifier(self):
        '''
        Add custom classifier to the model.
        '''
        average_pooling_2d = keras.layers.GlobalAveragePooling2D()(
            self.model.layers[-1].output)
        dropout = keras.layers.Dropout(0.2)(average_pooling_2d)
        dense = keras.layers.Dense(
            38, activation=tf.nn.softmax)(dropout)

        # Freeze the Model
        self.model.trainable = not self.transfer_learning

        model_with_classifier = keras.Model(
            inputs=self.model.inputs,
            outputs=[dense])

        self.model = model_with_classifier

    def __add_double_classifier(self):
        avgerage_pooling_2d = keras.layers.GlobalAveragePooling2D()(
            self.model.layers[-1].output)

        dropout1 = keras.layers.Dropout(0.15)(avgerage_pooling_2d)
        dense1 = keras.layers.Dense(
            14, activation=tf.nn.softmax, name='plant')(dropout1)

        concat = keras.layers.concatenate([avgerage_pooling_2d, dense1])

        dropout2 = keras.layers.Dropout(0.15)(concat)
        dense2 = keras.layers.Dense(
            21, activation=tf.nn.softmax, name='disease')(dropout2)

        self.model.trainable = not self.transfer_learning

        model_with_classifiers = keras.Model(
            inputs=self.model.inputs,
            outputs=[dense1, dense2]
        )

        self.model = model_with_classifiers
