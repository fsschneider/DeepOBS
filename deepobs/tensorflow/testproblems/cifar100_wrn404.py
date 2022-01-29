# -*- coding: utf-8 -*-
"""Wide ResNet 40-4 architecture for CIFAR-100."""

import tensorflow as tf

from ..datasets.cifar100 import cifar100
from ._wrn import _wrn
from .testproblem import TestProblem


class cifar100_wrn404(TestProblem):
    """DeepOBS test problem class for the Wide Residual Network 40-4 architecture\
    for CIFAR-100.

  Details about the architecture can be found in the `original paper`_.
  L2-Regularization is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  Training settings recommended in the `original paper`_:
  ``batch size = 128``, ``num_epochs = 200`` using the Momentum optimizer
  with :math:`\\mu = 0.9` and an initial learning rate of ``0.1`` with a decrease by
  ``0.2`` after ``60``, ``120`` and ``160`` epochs.

  .. _original paper: https://arxiv.org/abs/1605.07146

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
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

    def __init__(self, batch_size, l2_reg=0.0005):
        """Create a new WRN 40-4 test problem instance on Cifar-100.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(cifar100_wrn404, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the Wide ResNet 40-4 test problem on Cifar-100."""
        self.dataset = cifar100(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        training = tf.equal(self.dataset.phase, "train")
        x, y = self.dataset.batch
        linear_outputs = _wrn(
            x,
            training,
            num_residual_units=6,
            widening_factor=4,
            num_outputs=100,
            l2_reg=self._l2_reg,
        )

        self.losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=linear_outputs
        )
        y_pred = tf.argmax(input=linear_outputs, axis=1)
        y_correct = tf.argmax(input=y, axis=1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.compat.v1.losses.get_regularization_loss()
