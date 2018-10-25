# -*- coding: utf-8 -*-
"""
This is an implementation of the All-CNN-C model for CIFAR proposed in [1].
The paper does not comment on initialization; here we use Xavier for conv filters and constant 0.1 for biases.

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
    """"Class providing the functionality for the all convolutional architecture (All-CNN-C) from the `Striving for simplicity`_ paper on `CIFAR-100`.

    Details about the architecture can be found in the paper. The All-CNN-C network consits of multiple convolutional layers, with two dropout layers in between. The paper does not comment on initialization; here we use Xavier for conv filters and constant 0.1 for biases.

    Basis data augmentation (random crop, left-right flip, lighting augmentation) is done on the training images.

    The training setting in the paper were: Batch size of ``256``, weight decay of ``0.0005``, total training time of ``350`` epochs, with a base learning rate  of ``0.05`` and a decrease by factor ``10`` after ``200``, ``250`` and ``300`` epochs. Training was done using `momentum` with a momentum parameter of ``0.9``.

    Args:
        batch_size (int): Batch size of the data points. Defaults to ``128``.
        weight_decay (float): Weight decay factor, which is only applied to the weights and not the biases. Defaults to ``0.0005``.

    Attributes:
        data_loading (deepobs.data_loading): Data loading class for `CIFAR-100`, :class:`.cifar100_input.data_loading`.
        losses (tf.Tensor): Tensor of size ``batch_size`` containing the individual losses per data point.
        accuracy (tf.Tensor): Tensor containing the accuracy of the model.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval epoch.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase.

    .. _Striving for simplicity: https://arxiv.org/abs/1412.6806
    """
    def __init__(self, batch_size=128, weight_decay=0.0005):
        """Initializes the problem set_up class.

        Args:
            batch_size (int): Batch size of the data points. Defaults to ``128``.
            weight_decay (float): Weight decay factor. In this model weight decay is applied to the weights, but not the biases. Defaults to ``0.0005``.

        """
        self.data_loading = cifar100_input.data_loading(batch_size=batch_size)
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
            weight_decay (float): Weight decay factor, which is only applied to the weights and not the biases.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        X, y, phase = self.data_loading.load()
        print "X", X.get_shape()

        cond_keep_prob_1 = tf.cond(tf.equal(phase, tf.constant("train")),
                                   lambda: tf.constant(0.8),
                                   lambda: tf.constant(1.0))
        X_drop = tf.nn.dropout(X, keep_prob=cond_keep_prob_1)

        W_conv1 = self.conv_filter("W_conv1", [3, 3, 3, 96])
        b_conv1 = self.bias_variable("b_conv1", [96], init_val=0.1)
        h_conv1 = tf.nn.relu(self.conv2d(X_drop, W_conv1) + b_conv1)
        print "Shape of h_conv1", h_conv1.get_shape()

        W_conv2 = self.conv_filter("W_conv2", [3, 3, 96, 96])
        b_conv2 = self.bias_variable("b_conv2", [96], init_val=0.1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2) + b_conv2)
        print "Shape of h_conv2", h_conv2.get_shape()

        W_conv3 = self.conv_filter("W_conv3", [3, 3, 96, 96])
        b_conv3 = self.bias_variable("b_conv3", [96], init_val=0.1)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, stride=2) + b_conv3)
        print "Shape of h_conv3", h_conv3.get_shape()

        cond_keep_prob_2 = tf.cond(tf.equal(phase, tf.constant("train")),
                                   lambda: tf.constant(0.5),
                                   lambda: tf.constant(1.0))
        h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob=cond_keep_prob_2)

        W_conv4 = self.conv_filter("W_conv4", [3, 3, 96, 192])
        b_conv4 = self.bias_variable("b_conv4", [192], init_val=0.1)
        h_conv4 = tf.nn.relu(self.conv2d(h_conv3_drop, W_conv4) + b_conv4)
        print "Shape of h_conv4", h_conv4.get_shape()

        W_conv5 = self.conv_filter("W_conv5", [3, 3, 192, 192])
        b_conv5 = self.bias_variable("b_conv5", [192], init_val=0.1)
        h_conv5 = tf.nn.relu(self.conv2d(h_conv4, W_conv5) + b_conv5)
        print "Shape of h_conv5", h_conv5.get_shape()

        W_conv6 = self.conv_filter("W_conv6", [3, 3, 192, 192])
        b_conv6 = self.bias_variable("b_conv6", [192], init_val=0.1)
        h_conv6 = tf.nn.relu(self.conv2d(h_conv5, W_conv6, stride=2) + b_conv6)
        print "Shape of h_conv6", h_conv6.get_shape()

        h_conv6_drop = tf.nn.dropout(h_conv6, keep_prob=cond_keep_prob_2)

        W_conv7 = self.conv_filter("W_conv7", [3, 3, 192, 192])
        b_conv7 = self.bias_variable("b_conv7", [192], init_val=0.1)
        h_conv7 = tf.nn.relu(
            self.conv2d(h_conv6_drop, W_conv7, padding="VALID") + b_conv7)
        print "Shape of h_conv7", h_conv7.get_shape()

        W_conv8 = self.conv_filter("W_conv8", [1, 1, 192, 192])
        b_conv8 = self.bias_variable("b_conv8", [192], init_val=0.1)
        h_conv8 = tf.nn.relu(self.conv2d(h_conv7, W_conv8) + b_conv8)
        print "Shape of h_conv8", h_conv8.get_shape()

        W_conv9 = self.conv_filter("W_conv9", [1, 1, 192, 100])
        b_conv9 = self.bias_variable("b_conv9", [100], init_val=0.1)
        h_conv9 = tf.nn.relu(self.conv2d(h_conv8, W_conv9) + b_conv9)
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

    def conv_filter(self, name, shape):
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
