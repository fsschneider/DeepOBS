# -*- coding: utf-8 -*-
"""
This module implements the wide residual network (WRN) [1] architectures on the
CIFAR-100 data set. This is not a stand-alone deepobs test problem, but is
instantiated by the test problems cifar100_wrn404, et cetera.

The tensorflow code is adapted from [2].


[1]: https://arxiv.org/abs/1605.07146
[2]: https://github.com/dalgu90/wrn-tensorflow
"""

import numpy as np
import tensorflow as tf
import cifar100_input


class set_up:
    def __init__(self, batch_size, num_residual_units, k, weight_decay, bn_decay):
        self.data_loading = cifar100_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(
            num_residual_units, k, weight_decay, bn_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group(
            [self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, num_residual_units, k, weight_decay, bn_decay):

        # Number of filter channels and stride for the blocks
        filters = [16, 16 * k, 32 * k, 64 * k]
        strides = [1, 2, 2]

    #  num_residual_units = 4
    #  k = 1
    #  bn_averaging = 0.9

        X, y, phase = self.data_loading.load()

        # Initial convolution layer
        x = self._conv(X, filter_size=3, out_channels=16,
                       stride=1, name='conv_0')

        # Loop over three residual blocks
        for i in xrange(1, 4, 1):

            # First residual unit
            with tf.variable_scope('unit_%d_0' % i):
                x = self._batch_norm(
                    x, phase=phase, decay=bn_decay, name="bn_1")
                x = tf.nn.relu(x, name='relu_1')

                # Shortcut
                if filters[i - 1] == filters[i]:
                    if strides[i - 1] == 1:
                        shortcut = tf.identity(x)
                    else:
                        shortcut = tf.nn.max_pool(x, [1, strides[i - 1], strides[i - 1], 1],
                                                  [1, strides[i - 1], strides[i - 1], 1], 'VALID')
                else:
                    shortcut = self._conv(x, filter_size=1, out_channels=filters[i],
                                          stride=strides[i - 1], name='shortcut')

                # Residual
                x = self._conv(x, filter_size=3, out_channels=filters[i], stride=strides[i - 1],
                               name='conv_1')
                x = self._batch_norm(
                    x, phase=phase, decay=bn_decay, name="bn_2")
                x = tf.nn.relu(x, name='relu_2')
                x = self._conv(x, filter_size=3,
                               out_channels=filters[i], stride=1, name='conv_2')

                # Merge
                x = x + shortcut

            # further residual units
            for j in xrange(1, num_residual_units, 1):
                with tf.variable_scope('unit_%d_%d' % (i, j)):
                    # Shortcut
                    shortcut = x

                    # Residual
                    x = self._batch_norm(
                        x, phase=phase, decay=bn_decay, name="bn_1")
                    x = tf.nn.relu(x, name='relu_1')
                    x = self._conv(x, filter_size=3,
                                   out_channels=filters[i], stride=1, name='conv_1')
                    x = self._batch_norm(
                        x, phase=phase, decay=bn_decay, name="bn_2")
                    x = tf.nn.relu(x, name='relu_2')
                    x = self._conv(x, filter_size=3,
                                   out_channels=filters[i], stride=1, name='conv_2')

                    # Merge
                    x = x + shortcut

        # Last unit
        with tf.variable_scope('unit_last'):
            x = self._batch_norm(x, phase=phase, decay=bn_decay)
            x = tf.nn.relu(x, name="relu")
            x = tf.reduce_mean(x, [1, 2])

        # Reshaping and final fully-connected layer
        with tf.variable_scope('fully-connected'):
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1]])
            linear_outputs = self._fc(x, 100)

        # Softmax and loss
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs)

        # Add weight decay to the weight variables, but not to the biases
        for W in tf.get_collection("regularizable_variables"):
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay * tf.nn.l2_loss(W))

        # Compute mean accuracy
        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy

    def _batch_norm(self, x, phase, decay=0.9, name="batch_norm"):
        """Apply batch normalization to tensor x.

        Apply batch normalization to tensor x, switching between train and evaluation mode depending on the value of the placeholder phase ("train", "train_eval", "test").
        """
        with tf.variable_scope(name):
            # Compute the mean and variance of x across the axes 0, 1 and 2
            # TODO: with this axis reduction, this is GLOBAL normalization, is this what we want?
            mean_batch, variance_batch = tf.nn.moments(x, [0, 1, 2])

            # Allocate variables to maintain a moving average of the batch mean/variance
            mean_avg = tf.get_variable('mean_avg', mean_batch.get_shape(), tf.float32,
                                       initializer=tf.zeros_initializer, trainable=False)
            variance_avg = tf.get_variable('std_avg', variance_batch.get_shape(), tf.float32,
                                           initializer=tf.ones_initializer, trainable=False)

            # Allocate variables for the beta and gamma in batch norm
            # TODO: Do we want those to be trainable?
            beta = tf.get_variable('beta', mean_batch.get_shape(), tf.float32,
                                   initializer=tf.zeros_initializer, trainable=True)
            gamma = tf.get_variable('gamma', variance_batch.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer, trainable=True)

            # Add operations updating the moving averages of mean and variance
            # These ops are added to the UPDATE_OPS graph collection and must be added
            # as a dependency for the train step in order to be executed
            update_mean = mean_avg.assign(
                decay * mean_avg + (1.0 - decay) * mean_batch)
            update_variance = variance_avg.assign(
                decay * variance_avg + (1.0 - decay) * variance_batch)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)

            # Select batch mean/variance when phase=="train", otherwise select the
            # moving averages
            mean, variance = tf.cond(tf.equal(phase, "train"),
                                     lambda: (mean_batch, variance_batch),
                                     lambda: (mean_avg, variance_avg))

            # Return batch-normalized tensor
            return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)

    def _conv(self, x, filter_size, out_channels, stride, padding="SAME", name="conv"):
        """Apply a convolution to tensor ``x`` with a convolution kernel of shape
        ``filter_size * filter_size * out_channels``, as well as stride and padding
        as specified. The kernel is created/retrieved via tf.get_variable. No bias is added and no non-linearity is applied."""

        in_shape = x.get_shape()

        with tf.variable_scope(name):
            init = tf.random_normal_initializer(
                stddev=np.sqrt(1.0 / filter_size / filter_size / out_channels))
            W = tf.get_variable("W",
                                [filter_size, filter_size,
                                    in_shape[3], out_channels],
                                tf.float32,
                                initializer=init)
            if W not in tf.get_collection("regularizable_variables"):
                tf.add_to_collection("regularizable_variables", W)
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding, name="output")

    def _fc(self, x, out_dim, name='fc'):
        """Apply a affine transformation (fully-connected layer) to tensor ``x``
        with output dimension ``out_dim``. Weight matrix and bias vector are
        created/retrieved via tf.get_variable. No non-linearity is applied."""
        with tf.variable_scope(name):
            initializer = tf.random_normal_initializer(
                stddev=np.sqrt(1.0 / out_dim))
            W = tf.get_variable("W",
                                [x.get_shape()[1], out_dim],
                                tf.float32,
                                initializer=initializer)
            if W not in tf.get_collection("regularizable_variables"):
                tf.add_to_collection("regularizable_variables", W)

            b = tf.get_variable("b", [out_dim], tf.float32,
                                initializer=tf.constant_initializer(0.0))
            return tf.matmul(x, W) + b
