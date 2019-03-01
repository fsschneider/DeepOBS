# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:07:47 2018

@author: lballes
"""

import tensorflow as tf


def _wrn(x,
         training,
         num_residual_units,
         widening_factor,
         num_outputs,
         weight_decay,
         bn_momentum=0.9):
    def conv2d(inputs, filters, kernel_size, strides=1):
        """Convenience wrapper for conv layers."""
        return tf.layers.conv2d(
            inputs,
            filters,
            kernel_size,
            strides,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.initializers.glorot_uniform(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    def batch_normalization(inputs):
        """Convenience wrapper for batch norm."""
        return tf.layers.batch_normalization(
            inputs,
            axis=-1,
            momentum=bn_momentum,
            epsilon=1e-5,
            training=training)

    # Number of filter channels and stride for the blocks
    filters = [
        16, 16 * widening_factor, 32 * widening_factor, 64 * widening_factor
    ]
    strides = [1, 2, 2]

    # Initial convolution layer
    x = conv2d(x, 16, 3)

    # Loop over three residual blocks
    for i in range(1, 4, 1):

        # First residual unit
        with tf.variable_scope('unit_%d_0' % i):
            x = batch_normalization(x)
            x = tf.nn.relu(x)
            # Shortcut
            if filters[i - 1] == filters[i]:
                if strides[i - 1] == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.layers.max_pooling2d(x, strides[i - 1],
                                                       strides[i - 1])


#          shortcut = tf.nn.max_pool(x, [1, strides[i - 1], strides[i - 1], 1],
#                                    [1, strides[i - 1], strides[i - 1], 1], 'VALID')
            else:
                shortcut = conv2d(x, filters[i], 1, strides=strides[i - 1])
            # Residual
            x = conv2d(x, filters[i], 3, strides[i - 1])
            x = batch_normalization(x)
            x = tf.nn.relu(x)
            x = conv2d(x, filters[i], 3, 1)

            # Merge
            x = x + shortcut

        # further residual units
        for j in range(1, num_residual_units, 1):
            with tf.variable_scope('unit_%d_%d' % (i, j)):
                # Shortcut
                shortcut = x

                # Residual
                x = batch_normalization(x)
                x = tf.nn.relu(x)
                x = conv2d(x, filters[i], 3, 1)
                x = batch_normalization(x)
                x = tf.nn.relu(x)
                x = conv2d(x, filters[i], 3, 1)

                # Merge
                x = x + shortcut

    # Last unit
    with tf.variable_scope('unit_last'):
        x = batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, [1, 2])

    # Reshaping and final fully-connected layer
    with tf.variable_scope('fully-connected'):
        x_shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, x_shape[1]])
        linear_outputs = tf.layers.dense(
            x,
            num_outputs,
            kernel_initializer=tf.initializers.glorot_uniform(),
            bias_initializer=tf.initializers.constant(0.0),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    return linear_outputs
