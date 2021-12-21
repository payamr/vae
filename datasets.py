from typing import *

import numpy as np
import tensorflow as tf


def preprocess_images_mnist(images):
    """binarizes the images"""
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def load_mnist_images(load_labels: bool = False):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images_mnist(train_images)
    test_images = preprocess_images_mnist(test_images)
    if load_labels:
        return (train_images, train_labels), (test_images, test_labels)
    else:
        return train_images, test_images


def create_mnist_datasets(
        train_size=60000,
        batch_size=32,
        test_size=10000
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_images, test_images = load_mnist_images(load_labels=False)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)
    return train_dataset, test_dataset


def create_audio_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    raise NotImplementedError()

