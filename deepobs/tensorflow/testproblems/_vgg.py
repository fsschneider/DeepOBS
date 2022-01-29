# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:07:47 2018

@author: fschneider
"""

import tensorflow as tf


def _vgg(x, training, variant, num_outputs, l2_reg):
    def conv2d(inputs, filters, kernel_size=3, strides=(1, 1)):
        """Convenience wrapper for conv layers."""
        return tf.compat.v1.layers.conv2d(
            inputs,
            filters,
            kernel_size,
            strides,
            padding="same",
            activation=tf.nn.relu,
            bias_initializer=tf.compat.v1.initializers.constant(0.0),
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (l2_reg)),
        )

    def max_pool(inputs):
        """Convenience wrapper for max pool layers."""
        return tf.compat.v1.layers.max_pooling2d(
            inputs, pool_size=[2, 2], strides=[2, 2], padding="same"
        )

    # for now padd to 224x224 image size for VGG
    x = tf.image.resize(x, size=[224, 224])

    # conv1_1 and conv1_2
    x = conv2d(x, 64)
    x = conv2d(x, 64)
    x = max_pool(x)

    # conv2_1 and conv2_2
    x = conv2d(x, 128)
    x = conv2d(x, 128)
    x = max_pool(x)

    # conv3_1, conv3_2 and conv3_3 (and possibly conv3_4)
    x = conv2d(x, 256)
    x = conv2d(x, 256)
    x = conv2d(x, 256)
    if variant == 19:
        x = conv2d(x, 256)
    x = max_pool(x)

    # conv4_1, conv4_2 and conv4_3 (and possibly conv4_4)
    x = conv2d(x, 512)
    x = conv2d(x, 512)
    x = conv2d(x, 512)
    if variant == 19:
        x = conv2d(x, 512)
    x = max_pool(x)

    # conv5_1, conv5_2 and conv5_3 (and possibly conv5_4)
    x = conv2d(x, 512)
    x = conv2d(x, 512)
    x = conv2d(x, 512)
    if variant == 19:
        x = conv2d(x, 512)
    x = max_pool(x)
    x = tf.compat.v1.layers.flatten(x)

    # fc_1
    x = tf.compat.v1.layers.dense(x, 4096, activation=tf.nn.relu)
    x = tf.compat.v1.layers.dropout(x, rate=0.5, training=training)
    # fc_2
    x = tf.compat.v1.layers.dense(x, 4096, activation=tf.nn.relu)
    x = tf.compat.v1.layers.dropout(x, rate=0.5, training=training)
    # fc_3
    linear_outputs = tf.compat.v1.layers.dense(x, num_outputs)

    return linear_outputs
