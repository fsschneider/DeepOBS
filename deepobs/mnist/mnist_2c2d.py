# -*- coding: utf-8 -*-
"""
Vanilla CNN architecture adapted from the TensorFlow tutorial
(https://www.tensorflow.org/get_started/mnist/pros).
  - two conv layers with ReLUs, each followed by max-pooling
  - one fully-connected layers with ReLUs
  - 10-unit output layer with softmax
  - cross-entropy loss
  - no regularization
  - weight matrices initialized with truncated normal (stddev=0.05)
  - biases initialized to 0.05
"""

import tensorflow as tf
import mnist_input


class set_up:
    def __init__(self, batch_size=128, weight_decay=None):
        self.data_loading = mnist_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, weight_decay):
        if weight_decay is not None:
            print "WARNING: Weight decay is non-zero but no weight decay is used for this model."
        X, y, phase = self.data_loading.load()
        print "X", X.get_shape()

        W_conv1 = self._weight_variable("W_conv1", [5, 5, 1, 32])
        b_conv1 = self._bias_variable("b_conv1", [32], init_val=0.05)
        h_conv1 = tf.nn.relu(self._conv2d(X, W_conv1) + b_conv1)
        print "h_conv1", h_conv1.get_shape()

        h_pool1 = self._max_pool_2x2(h_conv1)
        print "h_pool1", h_pool1.get_shape()

        W_conv2 = self._weight_variable("W_conv2", [5, 5, 32, 64])
        b_conv2 = self._bias_variable("b_conv2", [64], init_val=0.05)
        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        print "h_conv2", h_conv2.get_shape()

        h_pool2 = self._max_pool_2x2(h_conv2)
        print "h_pool2", h_pool2.get_shape()

        dim = 7 * 7 * 64  # Shape of h_pool3 is [batch_size, 7, 7, 64]
        h_pool2_flat = tf.reshape(h_pool2, tf.stack([-1, dim]))
        print "h_pool2_flat", h_pool2_flat.get_shape()

        W_fc1 = self._weight_variable("W_fc1", [dim, 1024])
        b_fc1 = self._bias_variable("b_fc1", [1024], init_val=0.05)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = self._weight_variable("W_fc2", [1024, 10])
        b_fc2 = self._bias_variable("b_fc2", [10], init_val=0.05)
        linear_outputs = tf.matmul(h_fc1, W_fc2) + b_fc2

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=linear_outputs)

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy

    def _weight_variable(self, name, shape):
        init = tf.truncated_normal_initializer(stddev=0.05)
        return tf.get_variable(name, shape, initializer=init)

    def _bias_variable(self, name, shape, init_val):
        init = tf.constant_initializer(init_val)
        return tf.get_variable(name, shape, initializer=init)

    def _conv2d(self, x, W, stride=1, padding="SAME"):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
