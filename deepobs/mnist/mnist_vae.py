# -*- coding: utf-8 -*-
"""
Variational Autoencoder (VAE) on MNIST. Adapted from https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mnist_input


class set_up:
    """Class providing the functionality for a Variational Autoencoder (VAE) adapted from `here`_ on `MNIST`.

    Args:
        batch_size (int): Batch size of the data points. Defaults to ``64``.
        n_latent (int): Size of the latent space of the encoder. Defaults to ``8``.
        weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

    Attributes:
        data_loading (deepobs.data_loading): Data loading class for `MNIST`, :class:`.mnist_input.data_loading`.
        losses (tf.Tensor): Tensor of size ``batch_size`` containing the individual losses per data point.
        accuracy (tf.Tensor): Tensor containing the accuracy of the model. As there is no accuracy when the loss function is given directly, we set it to ``0``.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval epoch.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase.

    .. _here: https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776
    """
    def __init__(self, batch_size=64, n_latent=8, weight_decay=None):
        """Initializes the problem set_up class.

        Args:
            batch_size (int): Batch size of the data points. Defaults to ``64``.
            n_latent (int): Size of the latent space of the encoder. Defaults to ``8``.
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

        """
        self.data_loading = mnist_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(
            weight_decay=weight_decay, n_latent=n_latent)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group(
            [self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        """Returns the losses and the accuray of the model.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        return self.losses, self.accuracy

    def set_up(self, weight_decay=None, n_latent=8):
        """Sets up the test problem.

        Args:
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.
            n_latent (int): Size of the latent space of the encoder. Defaults to ``8``.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used for this model.")
        X, y, phase = self.data_loading.load()
        print("X", X.get_shape())
        X_flat = tf.reshape(X, shape=[-1, 28 * 28])

        sampled_z, mean, std = self.encoder(X, phase, n_latent=n_latent)
        img = self.decoder(sampled_z, phase, n_latent=n_latent)

        # Define Loss
        flatten_img = tf.reshape(img, [-1, 28 * 28])
        img_loss = tf.reduce_sum(tf.squared_difference(flatten_img, X_flat), 1)
        latent_loss = -0.5 * \
            tf.reduce_sum(1.0 + 2.0 * std - tf.square(mean) -
                          tf.exp(2.0 * std), 1)
        losses = img_loss + latent_loss

        # There is no accuracy here but keep it, so code can be reused
        accuracy = tf.zeros([1, 1], tf.float32)

        return losses, accuracy

    def lrelu(self, x, alpha=0.3):
        """Leaky ReLU activation function.

        Args:
            x (tf.Variable): Input to the activation function.
            alpha (float): Factor of the leaky ReLU. Defines how `leaky` it is. Defauylts to ``0.3``.

        Returns:
            tf.Variable: Output after the activation function.

        """
        return tf.maximum(x, tf.multiply(x, alpha))

    def encoder(self, X, phase, n_latent):
        """Encoder of the VAE. It consists of three convolutional and one dense layers. The convolutional layers use the leaky ReLU activation function. After each convolutional layer dropout is appleid with a keep probability of ``0.8``.

        Args:
            X (tf.Variable): Input to the encoder.
            phase (tf.Variable): Phase variable, determining if we are in training or evaluation mode.
            n_latent (int): Size of the latent space of the encoder. Defaults to ``8``.

        Returns:
            tupel: Output of the encoder, ``z``, the mean and the standard deviation.

        """
        cond_keep_prob_1 = tf.cond(tf.equal(phase, tf.constant("train")),
                                   lambda: tf.constant(0.8),
                                   lambda: tf.constant(1.0))
        activation = self.lrelu
        with tf.variable_scope("encoder", reuse=None):
            X = tf.reshape(X, shape=[-1, 28, 28, 1])
            x = tf.layers.conv2d(X, filters=64, kernel_size=4,
                                 strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, cond_keep_prob_1)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4,
                                 strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, cond_keep_prob_1)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4,
                                 strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, cond_keep_prob_1)
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units=n_latent)
            sd = 0.5 * tf.layers.dense(x, units=n_latent)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
            z = tf.add(mn, tf.multiply(epsilon, tf.exp(sd)), name="z")

            return z, mn, sd

    def decoder(self, sampled_z, phase, n_latent):
        """The decoder for the VAE. It uses two dense layers, followed by three deconvolutional layers (each with dropout= ``0.8``) a final dense layer. The dense layers use the leaky ReLU activation (except the last one, which uses softmax), while the deconvolutional ones use regular ReLU.

        Args:
            sampled_z (tf.Variable): Sampled ``z`` from the encoder of the size ``n_latent``.
            phase (tf.Variable): Phase variable, determining if we are in training or evaluation mode.
            n_latent (int): Size of the latent space of the encoder. Defaults to ``8``.

        Returns:
            tf.Variable: A tensor of the same size as the original images (``28`` by ``28``).

        """
        cond_keep_prob_1 = tf.cond(tf.equal(phase, tf.constant("train")),
                                   lambda: tf.constant(0.8),
                                   lambda: tf.constant(1.0))
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=24, activation=self.lrelu)
            x = tf.layers.dense(x, units=24 * 2 + 1, activation=self.lrelu)
            x = tf.reshape(x, [-1, 7, 7, 1])
            x = tf.layers.conv2d_transpose(
                x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, cond_keep_prob_1)
            x = tf.layers.conv2d_transpose(
                x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, cond_keep_prob_1)
            x = tf.layers.conv2d_transpose(
                x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
            img = tf.reshape(x, shape=[-1, 28, 28], name="decoder_op")
            return img

    def generate(self, sess, sampled_z=None):
        """Function to generate images using the decoder. Images are ploted directly.

        Args:
            sess (tf.Session): A TensorFlow session.
            sampled_z (tf.Variable): Sampled ``z`` with dimensions ``latent size`` times ``number of examples``. Defaults to ``None`` which uses five randomly sampled ``z`` from a normal with stddev = ``1.0``.

        """
        if sampled_z is None:
            sampled_z = [np.random.normal(0, 1, 8) for _ in range(5)]
        z = tf.get_default_graph().get_tensor_by_name("encoder/z:0")

        dec = tf.get_default_graph().get_tensor_by_name("decoder/decoder_op:0")
        imgs = sess.run(dec, feed_dict={z: sampled_z})
        imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

        for img in imgs:
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(img, cmap='gray')
            plt.show()
