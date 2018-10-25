# -*- coding: utf-8 -*-
"""
Multinomial logistic regression on MNIST.
  - no regularization
  - weights and biases initialized to 0
"""

import tensorflow as tf
import mnist_input


class set_up:
    """Class providing the functionality for multinomial logistic regression on `MNIST`.

    The model uses no regularization. All weights and biasses are initialized to be ``0``.

    Args:
        batch_size (int): Batch size of the data points. No default value specified.
        weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

    Attributes:
        data_loading (deepobs.data_loading): Data loading class for `MNIST`, :class:`.mnist_input.data_loading`.
        losses (tf.Tensor): Tensor of size ``batch_size`` containing the individual losses per data point.
        accuracy (tf.Tensor): Tensor containing the accuracy of the model.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval epoch.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase.

    """
    def __init__(self, batch_size, weight_decay=None):
        """Initializes the problem set_up class.

        Args:
            batch_size (int): Batch size of the data points. No default value specified.
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

        """
        self.data_loading = mnist_input.data_loading(batch_size=batch_size)
        self.losses, self.accuracy = self.set_up(weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op])
        self.test_init_op = tf.group([self.data_loading.test_init_op])

    def get(self):
        """Returns the losses and the accuray of the model.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        return self.losses, self.accuracy

    def set_up(self, weight_decay):
        """Sets up the test problem.

        Args:
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        if weight_decay is not None:
            print "WARNING: Weight decay is non-zero but no weight decay is used for this model."
        X, y, phase = self.data_loading.load()
        print "X", X.get_shape()

        X_flat = tf.reshape(X, [-1, 784])
        print "X_flat", X_flat.get_shape()

        W = tf.get_variable("W", [784, 10], initializer=tf.constant_initializer(0.0))
        b = tf.get_variable("b", [10], initializer=tf.constant_initializer(0.0))
        linear_outputs = tf.matmul(X_flat, W) + b
        print "linear_outputs", linear_outputs.get_shape()

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=linear_outputs)

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy
