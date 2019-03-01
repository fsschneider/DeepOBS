# -*- coding: utf-8 -*-
"""The all convolutional model All-CNN-C for CIFAR-100."""

import tensorflow as tf

from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem


class cifar100_allcnnc(TestProblem):
    """DeepOBS test problem class for the All Convolutional Neural Network C
  on Cifar-100.

  Details about the architecture can be found in the `original paper`_.

  The paper does not comment on initialization; here we use Xavier for conv
  filters and constant 0.1 for biases.

  A weight decay is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  .. _original paper: https://arxiv.org/abs/1412.6806

  The reference training parameters from the paper are ``batch size = 256``,
  ``num_epochs = 350`` using the Momentum optimizer with :math:`\\mu = 0.9` and
  an initial learning rate of :math:`\\alpha = 0.05` and decrease by a factor of
  ``10`` after ``200``, ``250`` and ``300`` epochs.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
        is used on the weights but not the biases.
        Defaults to ``5e-4``.

  Attributes:
    dataset: The DeepOBS data set class for Cifar-100.
    train_init_op: A tensorflow operation initializing the test problem for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the test problem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the test problem for
        evaluating on test data.
    losses: A tf.Tensor of shape (batch_size, ) containing the per-example loss
        values.
    regularizer: A scalar tf.Tensor containing a regularization term.
    accuracy: A scalar tf.Tensor containing the mini-batch mean accuracy.
  """

    def __init__(self, batch_size, weight_decay=0.0005):
        """Create a new All CNN C test problem instance on Cifar-100.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(cifar100_allcnnc, self).__init__(batch_size, weight_decay)

    def set_up(self):
        """Set up the All CNN C test problem on Cifar-100."""
        self.dataset = cifar100(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.test_init_op = self.dataset.test_init_op

        def conv2d(inputs, filters, kernel_size=3, strides=(1, 1), padding="same"):
            """Convenience wrapper for conv layers."""
            return tf.layers.conv2d(
                inputs,
                filters,
                kernel_size,
                strides,
                padding,
                activation=tf.nn.relu,
                bias_initializer=tf.initializers.constant(0.1),
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay))

        training = tf.equal(self.dataset.phase, "train")
        x, y = self.dataset.batch

        x = tf.layers.dropout(x, rate=0.2, training=training)

        x = conv2d(x, 96, 3)
        x = conv2d(x, 96, 3)
        x = conv2d(x, 96, 3, strides=(2, 2))

        x = tf.layers.dropout(x, rate=0.5, training=training)

        x = conv2d(x, 192, 3)
        x = conv2d(x, 192, 3)
        x = conv2d(x, 192, 3, strides=(2, 2))

        x = tf.layers.dropout(x, rate=0.5, training=training)

        x = conv2d(x, 192, 3, padding="valid")
        x = conv2d(x, 192, 1)
        x = conv2d(x, 100, 1)

        linear_outputs = tf.reduce_mean(x, axis=[1, 2])

        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs)

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.losses.get_regularization_loss()
