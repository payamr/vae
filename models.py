from typing import *

import numpy as np
import tensorflow as tf


def log_normal_pdf(sample, mean, logvar, reduce_axis=1):
    """log of a normal distribution with mean and ln(sigma**2)=logvar calculated at sample"""
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi) / 2., axis=reduce_axis)


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(
            self,
            latent_dim: int,
            filters: Sequence[int] = (32, 64),
            stride: int = 2,
            kernel: int = 3,
            input_shape: Tuple[Optional[int], Optional[int], int] = (28, 28, 1)
    ):
        super(CVAE, self).__init__()
        assert input_shape[0] % stride**len(filters) == 0
        assert input_shape[1] % stride**len(filters) == 0
        self.latent_dim = latent_dim
        self.filters = filters

        # encoder
        encoder_layers = [tf.keras.layers.InputLayer(input_shape=input_shape)]
        for f in filters:
            encoder_layers.append(
                tf.keras.layers.Conv2D(filters=f, kernel_size=kernel, strides=stride, activation='relu')
            )
        encoder_layers.extend([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),  # latent_dim number of means and stds each
        ])
        self.encoder = tf.keras.Sequential(encoder_layers)

        # decoder
        downsampled_res = (
            input_shape[0] // stride**(len(filters)),
            input_shape[1] // stride**(len(filters)),
        )
        decoder_layers = [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=np.prod(downsampled_res) * filters[-1], activation='relu'),
            tf.keras.layers.Reshape(target_shape=downsampled_res + (filters[-1], ))
        ]
        for f in filters[::-1]:
            decoder_layers.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=f, kernel_size=kernel, strides=stride, padding='same', activation='relu'
                ))
        # last layer
        decoder_layers.append(
            tf.keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=kernel, strides=1, padding='same')
        )
        self.decoder = tf.keras.Sequential(decoder_layers)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def call(self, x, training: bool):
        """expects input images, returns generated images"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z, apply_sigmoid=True)
        return output
