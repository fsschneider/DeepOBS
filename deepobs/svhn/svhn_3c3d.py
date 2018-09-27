# -*- coding: utf-8 -*-
"""
A vanilla CNN architecture for SVHN with
  - data augmentation (random crop, left-right flip, lighting augmentation)
    on the training images
  - three conv layers with ReLUs, each followed by max-pooling
  - two fully-connected layers with ReLUs
  - 10-unit output layer with softmax
  - cross-entropy loss
  - weight decay of 0.002 on the weights (not on biases)
  - weight matrices initialized with xavier_initializer
  - biases initialized to 0

Training settings
  - 40k steps at batch size 128
"""

import tensorflow as tf
import svhn_input


class set_up:
    def __init__(self, batch_size=128, weight_decay=0.002):
        self.data_loading = svhn_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, weight_decay):
        X, y, phase = self.data_loading.load()
        print "X", X.get_shape()

        W_conv1 = self._conv_filter("W_conv1", [5, 5, 3, 64])
        b_conv1 = self._bias_variable("b_conv1", [64], init_val=0.0)
        h_conv1 = tf.nn.relu(self._conv2d(X, W_conv1, padding="VALID") + b_conv1)
        print "h_conv1", h_conv1.get_shape()

        h_pool1 = self._max_pool_3x3(h_conv1)
        print "h_pool1", h_pool1.get_shape()

        W_conv2 = self._conv_filter("W_conv2", [3, 3, 64, 96])
        b_conv2 = self._bias_variable("b_conv2", [96], init_val=0.0)
        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2, padding="VALID") + b_conv2)
        print "h_conv2", h_conv2.get_shape()

        h_pool2 = self._max_pool_3x3(h_conv2)
        print "h_pool2", h_pool2.get_shape()

        W_conv3 = self._conv_filter("W_conv3", [3, 3, 96, 128])
        b_conv3 = self._bias_variable("b_conv3", [128], init_val=0.0)
        h_conv3 = tf.nn.relu(self._conv2d(h_pool2, W_conv3, padding="SAME") + b_conv3)
        print "h_conv3", h_conv3.get_shape()

        h_pool3 = self._max_pool_3x3(h_conv3)
        print "h_pool3", h_pool3.get_shape()

        dim = 1152  # Shape of h_pool3 is [batch_size, 3, 3, 128]
        h_pool3_flat = tf.reshape(h_pool3, tf.stack([-1, dim]))
        print "h_pool3_flat", h_pool3_flat.get_shape()

        W_fc1 = self._weight_matrix("W_fc1", [dim, 512])
        b_fc1 = self._bias_variable("b_fc1", [512], init_val=0.0)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        W_fc2 = self._weight_matrix("W_fc2", [512, 256])
        b_fc2 = self._bias_variable("b_fc2", [256], init_val=0.0)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        W_fc3 = self._weight_matrix("W_fc3", [256, 10])
        b_fc3 = self._bias_variable("b_fc3", [10], init_val=0.0)
        linear_outputs = tf.matmul(h_fc2, W_fc3) + b_fc3

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs)

        # Add weight decay to the weight variables, but not to the biases
        for W in [W_conv1, W_conv2, W_conv3, W_fc1, W_fc2, W_fc3]:
            tf.add_to_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay * tf.nn.l2_loss(W))

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy

    def _weight_matrix(self, name, shape):
        init = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape, initializer=init)

    def _conv_filter(self, name, shape):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.get_variable(name, shape, initializer=init)

    def _bias_variable(self, name, shape, init_val):
        init = tf.constant_initializer(init_val)
        return tf.get_variable(name, shape, initializer=init)

    def _conv2d(self, x, W, stride=1, padding="VALID"):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    def _max_pool_3x3(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
