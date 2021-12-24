import os
import time
from typing import *

import tensorflow as tf

from datasets import create_mnist_datasets, create_spoken_digit_spectrogram_dataset, spoken_digit_spectrograms
from models import CVAE


@tf.function
def train_step(model, x, optimizer, train_loss_object):
    """Executes one training step and returns the loss."""
    with tf.GradientTape() as tape:
        loss = model.loss(x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_object.update_state(loss)
    return loss


@tf.function
def test_step(model, x, test_loss_object):
    loss = model.loss(x)
    test_loss_object.update_state(loss)


def train_loop(train_dataset, test_dataset, params: Dict):

    train_summary_writer = tf.summary.create_file_writer(os.path.join(os.getcwd(), params['log_dir'], 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(os.getcwd(), params['log_dir'], 'val'))

    model_vae = CVAE(latent_dim=params['latent_dim'], input_shape=params['input_shape'])
    model_vae.compute_output_shape((params['batch_size'], ) + params['input_shape'])

    adam = tf.keras.optimizers.Adam(1e-4)

    num_batches_per_epoch = params.get('num_batches_per_epoch')
    num_validation_batches = params.get('num_validation_batches')
    tb_images_ever_n_epochs = params.get('tb_images_ever_n_epochs')

    for epoch in range(1, params['epochs'] + 1):
        start_time = time.time()
        train_loss_op = tf.keras.metrics.Mean(name='train_loss')
        test_loss_op = tf.keras.metrics.Mean(name='test_loss')
        for b, train_x in enumerate(train_dataset):
            train_step(model_vae, train_x, adam, train_loss_op)
            if num_validation_batches:
                if b >= num_batches_per_epoch:
                    break

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_op.result(), step=epoch)

        end_time = time.time()

        for b, test_x in enumerate(test_dataset):
            test_step(model_vae, test_x, test_loss_op)
            if num_validation_batches:
                if b >= num_validation_batches:
                    break

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss_op.result(), step=epoch)
            if tb_images_ever_n_epochs:
                if epoch % tb_images_ever_n_epochs == 0:
                    for test_x in test_dataset.take(1):  # take one batch for conditioning
                        gen_images = model_vae.call(test_x, training=False)
                        tf.summary.image(
                            f"generated_image", gen_images, max_outputs=params['num_tb_images'], step=epoch
                        )

        elbo = -test_loss_op.result()
        print(f'epoch: {epoch}, val_elbo: {elbo}, time elapse for current epoch: {end_time - start_time}')

        if epoch % params['checkpoint_every_n_epochs'] == 0:
            model_vae.save_weights(
                os.path.join(params['log_dir'], f"weights-epoch-{epoch:03d}"),
            )


if __name__ == '__main__':
    # # mnist
    # config = {
    #     'log_dir': os.path.join('/home/payam/Documents/log_dir', 'vae'),
    #     'latent_dim': 2,
    #     'input_shape': (28, 28, 1),
    #     'epochs': 40,
    #     'checkpoint_every_n_epochs': 2,
    #     'batch_size': 32,
    #     'num_tb_images': 8
    # }
    # train_data, test_data = create_mnist_datasets(batch_size=config['batch_size'])

    # spoken digits
    config = {
        'log_dir': os.path.join('/home/payam/Documents/log_dir', 'vae_spoken'),
        'latent_dim': 8,
        'input_shape': (64, 64, 4),
        'epochs': 40,
        'checkpoint_every_n_epochs': 4,
        'num_batches_per_epoch': 64,
        'num_validation_batches': 32,
        'batch_size': 16,
        'tb_images_ever_n_epochs': 4,
        'num_tb_images': 8,
        'buffer_size': 64,
    }
    train_data = create_spoken_digit_spectrogram_dataset(
        generator=spoken_digit_spectrograms(train=True),
        batch_size=config['batch_size'],
        buffer_size=config['buffer_size']
    )
    test_data = create_spoken_digit_spectrogram_dataset(
        generator=spoken_digit_spectrograms(train=False),
        batch_size=config['batch_size'],
        buffer_size=config['buffer_size']
    )

    train_loop(train_dataset=train_data, test_dataset=test_data, params=config)
