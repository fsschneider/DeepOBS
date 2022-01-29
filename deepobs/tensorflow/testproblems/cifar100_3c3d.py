# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for CIFAR-100."""

import tensorflow as tf

from ..datasets.cifar100 import cifar100
from ._3c3d import _3c3d
from .testproblem import TestProblem


class cifar100_3c3d(TestProblem):
    """DeepOBS test problem class for a three convolutional and three dense \
    layered neural network on Cifar-100.

  The network consists of

    - three conv layers with ReLUs, each followed by max-pooling
    - two fully-connected layers with ``512`` and ``256`` units and ReLU activation
    - 100-unit output layer with softmax
    - cross-entropy loss
    - L2 regularization on the weights (but not the biases) with a default
      factor of 0.002

  The weight matrices are initialized using Xavier initialization and the biases
  are initialized to ``0.0``.

  Args:
      batch_size (int): Batch size to use.
      l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
          is used on the weights but not the biases. Defaults to ``0.002``.

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

    def __init__(self, batch_size, l2_reg=0.002):
        """Create a new 3c3d test problem instance on Cifar-100.

        Args:
            batch_size (int): Batch size to use.
            l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
                is used on the weights but not the biases. Defaults to ``0.002``.
        """
        super(cifar100_3c3d, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the vanilla CNN test problem on Cifar-100."""
        self.dataset = cifar100(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        x, y = self.dataset.batch
        linear_outputs = _3c3d(x, num_outputs=100, l2_reg=self._l2_reg)

        self.losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=linear_outputs
        )

        y_pred = tf.argmax(input=linear_outputs, axis=1)
        y_correct = tf.argmax(input=y, axis=1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.compat.v1.losses.get_regularization_loss()
