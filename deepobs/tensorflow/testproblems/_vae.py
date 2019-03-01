# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:07:47 2018

@author: fschneider
"""

import tensorflow as tf


def _vae(x, training, n_latent=8):
    def conv2d(inputs, filters, kernel_size, strides, activation=tf.nn.relu):
        """Convenience wrapper for conv layers."""
        return tf.layers.conv2d(
            inputs,
            filters,
            kernel_size,
            strides,
            padding="same",
            activation=activation)

    def conv2d_transpose(inputs,
                         filters,
                         kernel_size,
                         strides,
                         activation=tf.nn.relu):
        """Convenience wrapper for conv layers."""
        return tf.layers.conv2d_transpose(
            inputs,
            filters,
            kernel_size,
            strides,
            padding='same',
            activation=activation)

    def lrelu(x, alpha=0.3):
        """Leaky ReLU activation function.

        Args:
            x (tf.Variable): Input to the activation function.
            alpha (float): Factor of the leaky ReLU. Defines how `leaky` it is.
                Defauylts to ``0.3``.

        Returns:
            tf.Variable: Output after the activation function.

        """
        return tf.maximum(x, tf.multiply(x, alpha))

    def encoder(x, training, n_latent=8):
        """Encoder of the VAE. It consists of three convolutional and one dense
        layers. The convolutional layers use the leaky ReLU activation function.
        After each convolutional layer dropout is appleid with a keep probability
        of ``0.8``.

        Args:
            x (tf.Variable): Input to the encoder.
            training (tf.Bool): Bool variable, determining if we are in
                training or evaluation mode.
            n_latent (int): Size of the latent space of the encoder.
                Defaults to ``8``.

        Returns:
            tupel: Output of the encoder, ``z``, the mean and the standard deviation.

        """
        with tf.variable_scope("encoder", reuse=None):
            x = tf.reshape(x, shape=[-1, 28, 28, 1])

            x = conv2d(x, 64, 4, 2, activation=lrelu)
            x = tf.layers.dropout(x, rate=0.2, training=training)

            x = conv2d(x, 64, 4, 2, activation=lrelu)
            x = tf.layers.dropout(x, rate=0.2, training=training)

            x = conv2d(x, 64, 4, 1, activation=lrelu)
            x = tf.layers.dropout(x, rate=0.2, training=training)

            x = tf.contrib.layers.flatten(x)

            mean = tf.layers.dense(x, units=n_latent)
            std_dev = 0.5 * tf.layers.dense(x, units=n_latent)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
            z = tf.add(mean, tf.multiply(epsilon, tf.exp(std_dev)), name="z")

            return z, mean, std_dev

    def decoder(sampled_z, training):
        """The decoder for the VAE. It uses two dense layers, followed by three
        deconvolutional layers (each with dropout= ``0.8``) a final dense layer.
        The dense layers use the leaky ReLU activation (except the last one,
        which uses softmax), while the deconvolutional ones use regular ReLU.

        Args:
            sampled_z (tf.Variable): Sampled ``z`` from the encoder of the size
                ``n_latent``.
            training (tf.Bool): Bool variable, determining if we are in
                training or evaluation mode.

        Returns:
            tf.Variable: A tensor of the same size as the original images
                (``28`` by ``28``).

        """
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=24, activation=lrelu)
            x = tf.layers.dense(x, units=24 * 2 + 1, activation=lrelu)

            x = tf.reshape(x, [-1, 7, 7, 1])

            x = conv2d_transpose(x, 64, 4, 2)
            x = tf.layers.dropout(x, rate=0.2, training=training)

            x = conv2d_transpose(x, 64, 4, 1)
            x = tf.layers.dropout(x, rate=0.2, training=training)

            x = conv2d_transpose(x, 64, 4, 1)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)

            img = tf.reshape(x, shape=[-1, 28, 28], name="decoder_op")

            return img

    x = tf.reshape(x, shape=[-1, 28 * 28])

    sampled_z, mean, std_dev = encoder(x, training, n_latent=n_latent)
    img = decoder(sampled_z, training)


    return img, mean, std_dev
