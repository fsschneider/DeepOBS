# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:07:47 2018

@author: fschneider
"""

import tensorflow as tf


def _mlp(x, num_outputs):
    def dense(inputs, units, activation=tf.nn.relu):
        """Convenience wrapper for max pool layers."""
        return tf.layers.dense(
            inputs,
            units,
            activation,
            bias_initializer=tf.initializers.constant(0.0),
            kernel_initializer=tf.truncated_normal_initializer(stddev=3e-2))

    x = tf.reshape(x, [-1, 784])

    x = dense(x, 1000)
    x = dense(x, 500)
    x = dense(x, 100)
    linear_outputs = dense(x, num_outputs, None)

    return linear_outputs
