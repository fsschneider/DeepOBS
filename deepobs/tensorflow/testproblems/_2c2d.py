# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:07:47 2018

@author: fschneider
"""

import tensorflow as tf


def _2c2d(x, num_outputs):
    def conv2d(inputs, filters):
        """Convenience wrapper for conv layers."""
        return tf.layers.conv2d(
            inputs,
            filters,
            kernel_size=5,
            padding="same",
            activation=tf.nn.relu,
            bias_initializer=tf.initializers.constant(0.05),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.05))

    def max_pool(inputs):
        """Convenience wrapper for max pool layers."""
        return tf.layers.max_pooling2d(
            inputs,
            pool_size=2,
            strides=2,
            padding='same',
        )

    def dense(inputs, units, activation):
        """Convenience wrapper for max pool layers."""
        return tf.layers.dense(
            inputs,
            units,
            activation,
            bias_initializer=tf.initializers.constant(0.05),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.05))

    x = conv2d(x, 32)
    x = max_pool(x)

    x = conv2d(x, 64)
    x = max_pool(x)

    x = tf.reshape(x, tf.stack([-1, 7 * 7 * 64]))

    x = dense(x, 1024, tf.nn.relu)

    linear_outputs = dense(x, num_outputs, None)

    return linear_outputs
