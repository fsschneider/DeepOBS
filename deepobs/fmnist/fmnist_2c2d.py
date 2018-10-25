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
import fmnist_input


class set_up:
    """Class providing the functionality for a vanilla CNN architecture on `Fashion-MNIST` adapted from the `TensorFlow tutorial`_ for `MNIST`.

    It consists of two convolutional layers with ReLU activations, each followed by max-pooling, followed by one fully-connected layer with ReLU activations and a 10-unit output layer with softmax. The model uses cross-entroy loss and no regularization. The weight matrices are initialized with truncated normal (stddev= ``0.05``) and the biases are initialized to ``0.05``.

    Args:
        batch_size (int): Batch size of the data points. Defaults to ``128``.
        weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

    Attributes:
        data_loading (deepobs.data_loading): Data loading class for `Fashion-MNIST`, :class:`.fmnist_input.data_loading`.
        losses (tf.Tensor): Tensor of size ``batch_size`` containing the individual losses per data point.
        accuracy (tf.Tensor): Tensor containing the accuracy of the model.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval epoch.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase.

    .. _TensorFlow tutorial: https://www.tensorflow.org/get_started/mnist/pros
    """
    def __init__(self, batch_size, weight_decay=None):
        """Initializes the problem set_up class.

        Args:
            batch_size (int): Batch size of the data points. Defaults to ``128``.
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

        """
        self.data_loading = fmnist_input.data_loading(batch_size=batch_size)
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

    def set_up(self, weight_decay=None):
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

        W_conv1 = self.weight_variable("W_conv1", [5, 5, 1, 32])
        b_conv1 = self.bias_variable("b_conv1", [32], init_val=0.05)
        h_conv1 = tf.nn.relu(self.conv2d(X, W_conv1) + b_conv1)
        print "h_conv1", h_conv1.get_shape()

        h_pool1 = self.max_pool_2x2(h_conv1)
        print "h_pool1", h_pool1.get_shape()

        W_conv2 = self.weight_variable("W_conv2", [5, 5, 32, 64])
        b_conv2 = self.bias_variable("b_conv2", [64], init_val=0.05)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        print "h_conv2", h_conv2.get_shape()

        h_pool2 = self.max_pool_2x2(h_conv2)
        print "h_pool2", h_pool2.get_shape()

        dim = 7 * 7 * 64  # Shape of h_pool3 is [batch_size, 7, 7, 64]
        h_pool2_flat = tf.reshape(h_pool2, tf.stack([-1, dim]))
        print "h_pool2_flat", h_pool2_flat.get_shape()

        W_fc1 = self.weight_variable("W_fc1", [dim, 1024])
        b_fc1 = self.bias_variable("b_fc1", [1024], init_val=0.05)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = self.weight_variable("W_fc2", [1024, 10])
        b_fc2 = self.bias_variable("b_fc2", [10], init_val=0.05)
        linear_outputs = tf.matmul(h_fc1, W_fc2) + b_fc2

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs)

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy

    def bias_variable(self, name, shape, init_val):
        """Creates a bias variable of given shape and initialized to a given value.

        Args:
            name (str): Name of the bias variable.
            shape (list): Dimensionality of the bias variable.
            init_val (float): Initial value of the bias variable.

        Returns:
            tf.Variable: Bias variable.

        """
        init = tf.constant_initializer(init_val)
        return tf.get_variable(name, shape, initializer=init)

    def conv2d(self, x, W, stride=1, padding="SAME"):
        """Creates a two dimensional convolutional layer on top of a given input.

        Args:
            x (tf.Variable): Input to the layer.
            W (tf.Variable): Weight variable of the convolutional layer.
            stride (int): Stride of the convolution. Defaults to ``1``.
            padding (str): Padding of the convolutional layers. Can be ``SAME`` or ``VALID``. Defaults to ``SAME``.

        Returns:
            tf.Variable: Output after the convolutional layer.

        """
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    def max_pool_2x2(self, x):
        """Creates a ``2`` by ``2`` max pool layer on top of a given input.

        Args:
            x (tf.Variable): Input to the layer.

        Returns:
            tf.Variable: Output after the max pool layer.

        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
