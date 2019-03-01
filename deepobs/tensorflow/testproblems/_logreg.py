# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:07:47 2018

@author: fschneider
"""

import tensorflow as tf


def _logreg(x, num_outputs):
    def dense(inputs, units):
        """Convenience wrapper for max pool layers."""
        return tf.layers.dense(
            inputs,
            units,
            activation=None,
            bias_initializer=tf.initializers.constant(0.0),
            kernel_initializer=tf.initializers.constant(0.0))

    x = tf.reshape(x, [-1, 784])

    linear_outputs = dense(x, num_outputs)

    return linear_outputs
