import os
from pathlib import Path

import tensorflow as tf

from .consts import DISEASE_CLASSES, PLANT_CLASSES


class Dataset:
    ''' Class to load and preprocess the dataset. 
    Loads images and labels from the given directory to tf.data.Dataset.


    Args:
        `data_dir (Path)`: Path to the dataset directory.
        `seed (int)`: Seed for shuffling the dataset.
        `repeat (int)`: Number of times to repeat the dataset.
        `shuffle_buffer_size (int)`: Size of the buffer for shuffling the dataset.
        `batch_size (int)`: Batch size for the dataset.
    '''

    def __init__(self,
                 data_dir: Path,
                 seed: int = 42,
                 repeat: int = 1,
                 shuffle_buffer_size: int = 10_000,
                 batch_size: int = 64) -> None:
        self.data_dir = data_dir
        self.seed = seed
        self.repeat = repeat
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size

        self.dataset = self.__load_dataset()\
            .shuffle(self.shuffle_buffer_size, seed=self.seed)\
            .repeat(self.repeat)\
            .batch(self.batch_size, drop_remainder=True)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def __load_dataset(self) -> tf.data.Dataset:
        # check if path has 'test' word in it
        dataset = tf.data.Dataset.list_files(str(self.data_dir / '*/*'))
        if 'test' in str(self.data_dir).lower():
            # file names issue - labels have camel case (regex?) and differs from the train/valid sets
            pass
        else:
            dataset = dataset.map(
                self.__preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def __get_labels(self, image_path):
        path = tf.strings.split(image_path, os.path.sep)[-2]
        plant = tf.strings.split(path, '___')[0]
        disease = tf.strings.split(path, '___')[1]

        one_hot_plant = plant == PLANT_CLASSES
        one_hot_disease = disease == DISEASE_CLASSES

        return tf.cast(one_hot_plant, dtype=tf.uint8, name=None), tf.cast(one_hot_disease, dtype=tf.uint8, name=None)

    def __get_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.cast(img, dtype=tf.float32, name=None) / 255.

    def __preprocess(self, image_path):
        labels = self.__get_labels(image_path)
        image = self.__get_image(image_path)

        # returns X, Y1, Y2
        return image, labels[0], labels[1]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)
