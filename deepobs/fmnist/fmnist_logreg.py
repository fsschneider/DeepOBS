# -*- coding: utf-8 -*-
"""
Multinomial logistic regression on Fashion-MNIST.
  - no regularization
  - weights and biases initialized to 0
"""

import tensorflow as tf
import fmnist_input


class set_up:
    def __init__(self, batch_size, weight_decay=None):
        self.data_loading = fmnist_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        return self.losses, self.accuracy

    def set_up(self, weight_decay=None):
        if weight_decay is not None:
            print "WARNING: Weight decay is non-zero but no weight decay is used for this model."
        X, y, phase = self.data_loading.load()
        print "X", X.get_shape()

        X_flat = tf.reshape(X, [-1, 784])
        print "X_flat", X_flat.get_shape()

        W = tf.get_variable(
            "W", [784, 10], initializer=tf.constant_initializer(0.0))
        b = tf.get_variable("b", [10], initializer=tf.constant_initializer(0.0))
        linear_outputs = tf.matmul(X_flat, W) + b
        print "linear_outputs", linear_outputs.get_shape()

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs)

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy
