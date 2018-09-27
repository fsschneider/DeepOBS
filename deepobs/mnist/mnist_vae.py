# -*- coding: utf-8 -*-
"""
Variational Autoencoder (VAE) on MNIST. Adapted from https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mnist_input


class set_up:
    def __init__(self, batch_size=64, n_latent=8, weight_decay=None):
        self.data_loading = mnist_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(
            weight_decay=weight_decay, n_latent=n_latent)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group(
            [self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, weight_decay=None, n_latent=8):
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
        return tf.maximum(x, tf.multiply(x, alpha))

    def encoder(self, X, phase, n_latent):
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

    def generate(self, sess, sampled_z=[np.random.normal(0, 1, 8) for _ in range(5)]):
        z = tf.get_default_graph().get_tensor_by_name("encoder/z:0")

        dec = tf.get_default_graph().get_tensor_by_name("decoder/decoder_op:0")
        imgs = sess.run(dec, feed_dict={z: sampled_z})
        imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

        for img in imgs:
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(img, cmap='gray')
            plt.show()
