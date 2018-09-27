# -*- coding: utf-8 -*-
"""
This is an implementation of the All-CNN-C model for CIFAR proposed in [1].
The paper does not comment on initialization; here we use Xavier for conv
filters and constant 0.1 for biases.

Reference training parameters from the paper:
- weight decay 0.0005
- SGD with momentum 0.9
- batch size 256
- base learning rate 0.05, decrease by factor 10 after 200, 250 and 300 epochs
  (thats approx 40k, 50k, 60k steps with batch size 256)
- total training time 350 epochs (approx 70k steps with batch size 256)

[1] Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014).
Striving for simplicity: The all convolutional net.
arXiv preprint arXiv:1412.6806.
"""

import tensorflow as tf
import cifar100_input


class set_up:
    def __init__(self, batch_size=128, weight_decay=0.0005):
        self.data_loading = cifar100_input.data_loading(batch_size=batch_size)
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

        cond_keep_prob_1 = tf.cond(tf.equal(phase, tf.constant("train")),
                                   lambda: tf.constant(0.8),
                                   lambda: tf.constant(1.0))
        X_drop = tf.nn.dropout(X, keep_prob=cond_keep_prob_1)

        W_conv1 = self._conv_filter("W_conv1", [3, 3, 3, 96])
        b_conv1 = self._bias_variable("b_conv1", [96], init_val=0.1)
        h_conv1 = tf.nn.relu(self._conv2d(X_drop, W_conv1) + b_conv1)
        print "Shape of h_conv1", h_conv1.get_shape()

        W_conv2 = self._conv_filter("W_conv2", [3, 3, 96, 96])
        b_conv2 = self._bias_variable("b_conv2", [96], init_val=0.1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2) + b_conv2)
        print "Shape of h_conv2", h_conv2.get_shape()

        W_conv3 = self._conv_filter("W_conv3", [3, 3, 96, 96])
        b_conv3 = self._bias_variable("b_conv3", [96], init_val=0.1)
        h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3, stride=2) + b_conv3)
        print "Shape of h_conv3", h_conv3.get_shape()

        cond_keep_prob_2 = tf.cond(tf.equal(phase, tf.constant("train")),
                                   lambda: tf.constant(0.5),
                                   lambda: tf.constant(1.0))
        h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob=cond_keep_prob_2)

        W_conv4 = self._conv_filter("W_conv4", [3, 3, 96, 192])
        b_conv4 = self._bias_variable("b_conv4", [192], init_val=0.1)
        h_conv4 = tf.nn.relu(self._conv2d(h_conv3_drop, W_conv4) + b_conv4)
        print "Shape of h_conv4", h_conv4.get_shape()

        W_conv5 = self._conv_filter("W_conv5", [3, 3, 192, 192])
        b_conv5 = self._bias_variable("b_conv5", [192], init_val=0.1)
        h_conv5 = tf.nn.relu(self._conv2d(h_conv4, W_conv5) + b_conv5)
        print "Shape of h_conv5", h_conv5.get_shape()

        W_conv6 = self._conv_filter("W_conv6", [3, 3, 192, 192])
        b_conv6 = self._bias_variable("b_conv6", [192], init_val=0.1)
        h_conv6 = tf.nn.relu(self._conv2d(h_conv5, W_conv6, stride=2) + b_conv6)
        print "Shape of h_conv6", h_conv6.get_shape()

        h_conv6_drop = tf.nn.dropout(h_conv6, keep_prob=cond_keep_prob_2)

        W_conv7 = self._conv_filter("W_conv7", [3, 3, 192, 192])
        b_conv7 = self._bias_variable("b_conv7", [192], init_val=0.1)
        h_conv7 = tf.nn.relu(
            self._conv2d(h_conv6_drop, W_conv7, padding="VALID") + b_conv7)
        print "Shape of h_conv7", h_conv7.get_shape()

        W_conv8 = self._conv_filter("W_conv8", [1, 1, 192, 192])
        b_conv8 = self._bias_variable("b_conv8", [192], init_val=0.1)
        h_conv8 = tf.nn.relu(self._conv2d(h_conv7, W_conv8) + b_conv8)
        print "Shape of h_conv8", h_conv8.get_shape()

        W_conv9 = self._conv_filter("W_conv9", [1, 1, 192, 100])
        b_conv9 = self._bias_variable("b_conv9", [100], init_val=0.1)
        h_conv9 = tf.nn.relu(self._conv2d(h_conv8, W_conv9) + b_conv9)
        print "Shape of h_conv9", h_conv9.get_shape()

        linear_outputs = tf.reduce_mean(h_conv9, axis=[1, 2])
        print "Shape of linear_outputs", linear_outputs.get_shape()

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                            logits=linear_outputs)

        # Add weight decay to the weight variables, but not to the biases
        for W in [W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_conv6, W_conv7, W_conv8, W_conv9]:
            tf.add_to_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay * tf.nn.l2_loss(W))

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy

    def _conv_filter(self, name, shape):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.get_variable(name, shape, initializer=init)

    def _bias_variable(self, name, shape, init_val):
        init = tf.constant_initializer(init_val)
        return tf.get_variable(name, shape, initializer=init)

    def _conv2d(self, x, W, stride=1, padding="SAME"):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
