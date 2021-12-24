from typing import *

import hub
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


def spoken_digit_spectrograms(train: bool, test_ratio: float = 0.1) -> Callable:
    """
    generator which yields normalized spectrograms (type=np.uint8)
    """
    assert 0 < test_ratio < 1
    ds = hub.load("hub://activeloop/spoken_mnist")
    test_period = int(1. / test_ratio)

    def skip_condition(counter) -> bool:
        return counter % test_period == 0 if train else counter % test_period != 0

    def _gen():
        for i, sample in enumerate(ds):
            if skip_condition(i):
                continue
            spec = sample['spectrograms'].numpy().astype(np.float32) / 255.
            yield spec

    return _gen


def create_spoken_digit_spectrogram_dataset(
        generator,
        batch_size: int,
        buffer_size: int = 10,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=tf.float32,
        output_shapes=(64, 64, 4)
    )
    return dataset.padded_batch(batch_size).shuffle(buffer_size).repeat().prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    test_data = create_spoken_digit_spectrogram_dataset(
        generator=spoken_digit_spectrograms(train=False),
        batch_size=3,
        buffer_size=2
    )
    for d in test_data.take(1):
        print('here')
        print(d)
