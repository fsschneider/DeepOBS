# -*- coding: utf-8 -*-
"""
A vanilla CNN architecture for CIFAR with
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
import cifar10_input


class set_up:
    """Class providing the functionality for a vanilla CNN architecture on `CIFAR-10`.

    It consists of three convolutional layers with ReLU activations, each followed by max-pooling, followed by two fully-connected layer with ReLU activations and a 10-unit output layer with softmax. The model uses cross-entroy loss. A weight decay is used on the weights (but not the biases) which defaults to ``0.002``. The weight matrices are initialized using the `Xavier-Initializer` and the biases are initialized to ``0``.

    Basis data augmentation (random crop, left-right flip, lighting augmentation) is done on the training images.

    A suggested training settings is for ``100`` epochs with a batch size of ``128``.

    Args:
        batch_size (int): Batch size of the data points. Defaults to ``128``.
        weight_decay (float): Weight decay factor. In this model weight decay is applied to the weights, but not the biases. Defaults to ``0.002``.

    Attributes:
        data_loading (deepobs.data_loading): Data loading class for `CIFAR-10`, :class:`.cifar10_input.data_loading`.
        losses (tf.Tensor): Tensor of size ``batch_size`` containing the individual losses per data point.
        accuracy (tf.Tensor): Tensor containing the accuracy of the model.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval epoch.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase.
    """
    def __init__(self, batch_size=128, weight_decay=0.002):
        """Initializes the problem set_up class.

        Args:
            batch_size (int): Batch size of the data points. Defaults to ``128``.
            weight_decay (float): Weight decay factor. In this model weight decay is applied to the weights, but not the biases. Defaults to ``0.002``.

        """
        self.data_loading = cifar10_input.data_loading(batch_size=batch_size)
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
            weight_decay (float): Weight decay factor. In this model weight decay is applied to the weights, but not the biases.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        X, y, phase = self.data_loading.load()
        print "X", X.get_shape()

        W_conv1 = self._conv_filter("W_conv1", [5, 5, 3, 64])
        b_conv1 = self.bias_variable("b_conv1", [64], init_val=0.0)
        h_conv1 = tf.nn.relu(self.conv2d(X, W_conv1, padding="VALID") + b_conv1)
        print "h_conv1", h_conv1.get_shape()

        h_pool1 = self.max_pool_3x3(h_conv1)
        print "h_pool1", h_pool1.get_shape()

        W_conv2 = self._conv_filter("W_conv2", [3, 3, 64, 96])
        b_conv2 = self.bias_variable("b_conv2", [96], init_val=0.0)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, padding="VALID") + b_conv2)
        print "h_conv2", h_conv2.get_shape()

        h_pool2 = self.max_pool_3x3(h_conv2)
        print "h_pool2", h_pool2.get_shape()

        W_conv3 = self._conv_filter("W_conv3", [3, 3, 96, 128])
        b_conv3 = self.bias_variable("b_conv3", [128], init_val=0.0)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, padding="SAME") + b_conv3)
        print "h_conv3", h_conv3.get_shape()

        h_pool3 = self.max_pool_3x3(h_conv3)
        print "h_pool3", h_pool3.get_shape()

        dim = 1152  # Shape of h_pool3 is [batch_size, 3, 3, 128]
        h_pool3_flat = tf.reshape(h_pool3, tf.stack([-1, dim]))
        print "h_pool3_flat", h_pool3_flat.get_shape()

        W_fc1 = self.weight_matrix("W_fc1", [dim, 512])
        b_fc1 = self.bias_variable("b_fc1", [512], init_val=0.0)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        W_fc2 = self.weight_matrix("W_fc2", [512, 256])
        b_fc2 = self.bias_variable("b_fc2", [256], init_val=0.0)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        W_fc3 = self.weight_matrix("W_fc3", [256, 10])
        b_fc3 = self.bias_variable("b_fc3", [10], init_val=0.0)
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

    def weight_matrix(self, name, shape):
        """Creates a weight matrix, initialized by the `Xavier-initializer`.

        Args:
            name (str): Name of the weight variable.
            shape (list): Dimensionality of the weight variable.

        Returns:
            tf.Variable: Weight variable.

        """
        init = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape, initializer=init)

    def _conv_filter(self, name, shape):
        """Creates a convolutional filter matrix, initialized by the `Xavier-initializer`.

        Args:
            name (str): Name of the filter variable.
            shape (list): Dimensionality of the filter variable.

        Returns:
            tf.Variable: Filter variable.

        """
        init = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.get_variable(name, shape, initializer=init)

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

    def conv2d(self, x, W, stride=1, padding="VALID"):
        """Creates a two dimensional convolutional layer on top of a given input.

        Args:
            x (tf.Variable): Input to the layer.
            W (tf.Variable): Weight variable of the convolutional layer.
            stride (int): Stride of the convolution. Defaults to ``1``.
            padding (str): Padding of the convolutional layers. Can be ``SAME`` or ``VALID``. Defaults to ``VALID``.

        Returns:
            tf.Variable: Output after the convolutional layer.

        """
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    def max_pool_3x3(self, x):
        """Creates a ``3`` by ``3`` max pool layer on top of a given input.

        Args:
            x (tf.Variable): Input to the layer.

        Returns:
            tf.Variable: Output after the max pool layer.

        """
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
