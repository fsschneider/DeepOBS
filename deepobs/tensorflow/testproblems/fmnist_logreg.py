# -*- coding: utf-8 -*-
"""Multinomial logistic regression on Fashion-MNIST."""

import tensorflow as tf

from ..datasets.fmnist import fmnist
from ._logreg import _logreg
from .testproblem import TestProblem


class fmnist_logreg(TestProblem):
    """DeepOBS test problem class for multinomial logistic regression on \
    Fasion-MNIST.

  No regularization is used and the weights and biases are initialized to ``0.0``.

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): No L2-Regularization (weight decay) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

  Attributes:
    dataset: The DeepOBS data set class for Fashion-MNIST.
    train_init_op: A tensorflow operation initializing the test problem for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the test problem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the test problem for
        evaluating on test data.
    losses: A tf.Tensor of shape (batch_size, ) containing the per-example loss
        values.
    regularizer: A scalar tf.Tensor containing a regularization term.
        Will always be ``0.0`` since no regularizer is used.
    accuracy: A scalar tf.Tensor containing the mini-batch mean accuracy.
  """

    def __init__(self, batch_size, l2_reg=None):
        """Create a new logistic regression test problem instance on Fashion-MNIST.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): No L2-Regularization (weight decay) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(fmnist_logreg, self).__init__(batch_size, l2_reg)

        if l2_reg is not None:
            print(
                "WARNING: L2-Regularization is non-zero but no L2-regularization is used",
                "for this model.",
            )

    def set_up(self):
        """Set up the logistic regression test problem on Fashion-MNIST."""
        self.dataset = fmnist(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        x, y = self.dataset.batch
        linear_outputs = _logreg(x, num_outputs=10)

        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs
        )

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.losses.get_regularization_loss()
