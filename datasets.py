from typing import *
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf
import tensorflow_datasets as tfds

import dsp.dsp as dsp

'''
MNIST image dataset (handwritten digits)
'''

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

'''
Spoken digit audio dataset
'''

def _preprocess_spoken_digit_audio_tf(example, with_labels):
    audio = example['audio']
    audio = tf.cast(audio, tf.float32)
        
    # remove DC offset
    audio_mean = tf.math.reduce_mean(audio, axis=0) 
    audio -= audio_mean
    
    # convert from int16 to float32 [-1.0, 1.0]
    # detected clipping using just 'tf.int16.max' to normalize, hence the '+ 1.0'...
    audio /= (tf.int16.max + 1.0) 
    
    if with_labels:
        return (audio, example['label'])
    return audio


def _load_spoken_digit_dataset_tf(percent_load, load_labels, download=False):
    # this dataset only has 'train' split
    all_data = tfds.load(name='spoken_digit',
                         split=[f'train[:{percent_load}%]'],
                         shuffle_files=False,
                         as_supervised=False,
                         download=download)

    all_data = all_data[0]
    all_data = all_data.map(lambda x: _preprocess_spoken_digit_audio_tf(x, load_labels), 
                            num_parallel_calls=tf.data.AUTOTUNE,
                            deterministic=True)
    return all_data


def convert_raw_spoken_digit_dataset(destination_dir, percent_load=100, download=False):
    if not destination_dir.is_dir():
        assert(destination_dir.parent.is_dir())
        Path.mkdir(destination_dir)

    all_data = _load_spoken_digit_dataset_tf(percent_load, True, download=download)
    total_count = len(all_data)
    print(f'Writing {total_count} examples as numpy arrays')
    
    # deterministic shuffling now so later train / test split is consistent 
    # but not biased by original order of raw dataset
    all_data = all_data.shuffle(total_count, seed=0)

    for example_id, (example_audio, example_label) in enumerate(all_data):
        destination_path = destination_dir / f'example_{example_id}.npy'
        audio = example_audio.numpy()
        label = int(example_label)
        example_type = np.dtype([('label', np.int8), ('audio', np.float32, (len(audio),))])
        example_array = np.rec.array((label, audio), dtype=example_type)
        np.save(destination_path, example_array)

def _get_spoken_digit_signature_and_padded_shape(config, load_labels):
    dsp_config = config['dsp']
    window_size = dsp_config['window_size']
    padding_factor = dsp_config['padding_factor']
    feature_names = dsp_config['feature_names']
    
    signature = {}
    padded_shapes = {}
    for feature_name in feature_names:
        feature_shape = dsp.get_stft_feature_shape(feature_name, window_size, padding_factor)
        signature[feature_name] = tf.TensorSpec(shape=feature_shape, dtype=tf.float32)
        padded_shapes[feature_name] = feature_shape
        
    signature = [signature]
        
    if load_labels:
        signature.append(tf.TensorSpec(shape=(), dtype=tf.int8))
        padded_shapes = (padded_shapes, ())
    
    return tuple(signature), padded_shapes


def _extract_spoken_digit_features_npy(example, config, load_labels):
    '''
    Computes requested features from raw audio data. Time corresponds to axis=0.
    '''
    audio = example.audio # this should be a numpy array
    label = example.label if load_labels else None
    # TODO data augmentation

    dsp_config = config['dsp']
    window = dsp_config['window']
    hop_size = dsp_config['hop_size']
    window_size = dsp_config['window_size']
    padding_factor = dsp_config['padding_factor']
    zero_phase = dsp_config['zero_phase']
    sqrt_window = dsp_config['sqrt_window']
    feature_names = dsp_config['feature_names']
    
    windowed_frame_data = dsp.windowed_frame_analysis(audio, window, hop_size, 
        padding_factor, zero_phase, sqrt_window)
    features = dsp.stft_analysis(windowed_frame_data['frames'], feature_names)
    
    # convert numpy arrays to tensors
    for feature_name in feature_names:
        feature_shape = dsp.get_stft_feature_shape(feature_name, window_size, padding_factor)
        feature_frames = features[feature_name].view(f'({feature_shape[2]},)float')
        features[feature_name] = tf.convert_to_tensor(feature_frames, dtype=tf.float32)
    
    if label is not None:
        label = tf.convert_to_tensor(label, dtype=tf.int8)
        return (features, label)
    
    return features


def _count_spoken_digit_dataset_npy(data_dir):
    return sum([1 for _ in data_dir.glob('*.npy')])


def _load_spoken_digit_dataset_npy(data_dir, config, load_labels):
    for data_file in data_dir.glob('*.npy'):
        data = np.load(data_file)
        yield _extract_spoken_digit_features_npy(np.rec.array(data, dtype=data.dtype), config, load_labels)


def load_spoken_digit_dataset(data_dir, config, load_labels):
    count = _count_spoken_digit_dataset_npy(data_dir)
    generator = lambda: _load_spoken_digit_dataset_npy(data_dir, config, load_labels)
    out_signature, padded_shapes = _get_spoken_digit_signature_and_padded_shape(config, load_labels)
    all_data = tf.data.Dataset.from_generator(generator, output_signature=out_signature)
    all_data = all_data.apply(tf.data.experimental.assert_cardinality(count))
    return all_data, padded_shapes


def create_spoken_digit_datasets(
        data_dir,
        config,
        batch_size=32,
        train_test_split=0.8,
        load_labels=False,
        repeat=True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:    
    all_data, padded_shapes = load_spoken_digit_dataset(data_dir, config, load_labels)
    total_count = len(all_data)

    train_count = int(train_test_split * total_count)
    
    train_data = all_data.take(train_count).cache().shuffle(train_count, reshuffle_each_iteration=True)
    test_data = all_data.skip(train_count).cache()

    if repeat == True:
        train_data = train_data.repeat()
        test_data = test_data.repeat()

    train_data = train_data.padded_batch(batch_size, padded_shapes=padded_shapes)
    test_data = test_data.padded_batch(batch_size, padded_shapes=padded_shapes)
    
    return (train_data.prefetch(tf.data.AUTOTUNE), test_data.prefetch(tf.data.AUTOTUNE))


if __name__ == '__main__':
    mode = 'audio' # 'image'

    if mode == 'image':
        train_data, test_data = create_mnist_datasets()

        for d in test_data.take(1):
            print('Example data:')
            print(d)
    elif mode == 'audio':
        data_dir = Path('./spoken_word_data')

        # load raw audio from TF dataset and convert to numpy arrays in .npy files:
        # (make sure to set download=True if you don't have the TFDS yet)
        convert_raw_spoken_digit_dataset(data_dir, 100, download=False)

        # load config file, load numpy dataset, extract audio features, generate TF dataset:
        config_path = Path('./vae_spoken_digit_config.yml')
        config = None
        with open(config_path, 'rt') as config_file:
            config = yaml.safe_load(config_file)
        load_labels = True

        window_type = config['dsp']['window_type']
        window_size = config['dsp']['window_size']

        window, hop_size = dsp.get_cola_window(window_type, window_size)
        config['dsp']['window'] = window
        config['dsp']['hop_size'] = hop_size

        train_data, test_data = create_spoken_digit_datasets(data_dir, config, load_labels=load_labels)
        
        for d in test_data.take(1):
            print('Example data:')
            print(d)
