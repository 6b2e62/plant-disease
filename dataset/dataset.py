import os
from pathlib import Path

import tensorflow as tf

from .consts import ALL_CLASSES, DISEASE_CLASSES, PLANT_CLASSES
import model


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
                 model_class,
                 data_dir: Path,
                 seed: int = 42,
                 repeat: int = 1,
                 # For now setting shuffle_buffer_size to smaller number due to system RAM issues on Google Colab
                 shuffle_buffer_size: int = 1000,
                 batch_size: int = 64) -> None:
        self.data_dir = data_dir
        self.seed = seed
        self.repeat = repeat
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.model_class = model_class

        self.dataset = self.__load_dataset()\
            .shuffle(self.shuffle_buffer_size, seed=self.seed)\
            .repeat(self.repeat)\
            .batch(self.batch_size, drop_remainder=True)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def get_dataset(self) -> tf.data.Dataset:
        return self.dataset

    def __load_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.list_files(str(self.data_dir / '*/*'))
        dataset = dataset.map(
            # For now setting num_parallel_calls to 2 due to system RAM issues on Google Colab
            self.__preprocess_all_in_one, num_parallel_calls=2)

        return dataset

    def __get_labels(self, image_path):
        path = tf.strings.split(image_path, os.path.sep)[-2]
        plant = tf.strings.split(path, '___')[0]
        disease = tf.strings.split(path, '___')[1]

        one_hot_plant = plant == PLANT_CLASSES
        one_hot_disease = disease == DISEASE_CLASSES

        return tf.cast(one_hot_plant, dtype=tf.uint8, name=None), tf.cast(one_hot_disease, dtype=tf.uint8, name=None)

    def __get_label(self, image_path):
        path = tf.strings.split(image_path, os.path.sep)[-2]
        one_hot = path == ALL_CLASSES

        return tf.cast(one_hot, dtype=tf.uint8, name=None)

    def __get_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.cast(img, dtype=tf.float32)
        if self.model_class == model.mobilenetv2.MobilenetV2Model:
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        elif self.model_class == model.resnet50v2.Resnet50V2Model:
            img = tf.keras.applications.resnet_v2.preprocess_input(img)
        elif self.model_class == model.efficentnetv2b0.EfficientNetV2B0Model:
            img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        else:
            img = img / 255.
        return img

    def __preprocess_all_in_one(self, image_path):
        label = self.__get_label(image_path)
        image = self.__get_image(image_path)
        return image, label

    def __preprocess(self, image_path):
        labels = self.__get_labels(image_path)
        image = self.__get_image(image_path)

        # returns X, Y1, Y2
        return image, labels[0], labels[1]

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)
