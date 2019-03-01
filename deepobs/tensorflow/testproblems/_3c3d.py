# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:07:47 2018

@author: fschneider
"""

import tensorflow as tf


def _3c3d(x, num_outputs, weight_decay):
    def conv2d(inputs, filters, kernel_size=3, padding="same"):
        """Convenience wrapper for conv layers."""
        return tf.layers.conv2d(
            inputs,
            filters,
            kernel_size,
            (1, 1),
            padding,
            activation=tf.nn.relu,
            bias_initializer=tf.initializers.constant(0.0),
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    def max_pool(inputs):
        """Convenience wrapper for max pool layers."""
        return tf.layers.max_pooling2d(
            inputs,
            pool_size=3,
            strides=2,
            padding='same',
        )

    def dense(inputs, units, activation):
        """Convenience wrapper for max pool layers."""
        return tf.layers.dense(
            inputs,
            units,
            activation,
            kernel_initializer=tf.initializers.glorot_uniform(),
            bias_initializer=tf.initializers.constant(0.0),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    x = conv2d(x, 64, 5, "valid")
    x = max_pool(x)

    x = conv2d(x, 96, 3, "valid")
    x = max_pool(x)

    x = conv2d(x, 128, 3, "same")
    x = max_pool(x)

    x = tf.reshape(x, tf.stack([-1, 3 * 3 * 128]))

    x = dense(x, 512, tf.nn.relu)

    x = dense(x, 256, tf.nn.relu)

    linear_outputs = dense(x, num_outputs, None)

    return linear_outputs
