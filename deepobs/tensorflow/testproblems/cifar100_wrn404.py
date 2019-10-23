# -*- coding: utf-8 -*-
"""Wide ResNet 40-4 architecture for CIFAR-100."""

import tensorflow as tf

from ._wrn import _wrn
from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem


class cifar100_wrn404(TestProblem):
    """DeepOBS test problem class for the Wide Residual Network 40-4 architecture\
    for CIFAR-100.

  Details about the architecture can be found in the `original paper`_.
  A weight decay is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  Training settings recommenden in the `original paper`_:
  ``batch size = 128``, ``num_epochs = 200`` using the Momentum optimizer
  with :math:`\\mu = 0.9` and an initial learning rate of ``0.1`` with a decrease by
  ``0.2`` after ``60``, ``120`` and ``160`` epochs.

  .. _original paper: https://arxiv.org/abs/1605.07146

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
        """Create a new WRN 40-4 test problem instance on Cifar-100.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(cifar100_wrn404, self).__init__(batch_size, weight_decay)

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
            weight_decay=self._weight_decay,
        )

        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs
        )
        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.losses.get_regularization_loss()
