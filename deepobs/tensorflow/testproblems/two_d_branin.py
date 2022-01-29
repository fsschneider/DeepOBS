# -*- coding: utf-8 -*-
"""A simple 2D Noisy Branin Loss Function."""

import numpy as np

import tensorflow as tf

from ..datasets.two_d import two_d
from .testproblem import TestProblem


class two_d_branin(TestProblem):
    r"""DeepOBS test problem class for a stochastic version of the\
    two-dimensional Branin function as the loss function.

    Using the deterministic `Branin function
    <https://www.sfu.ca/~ssurjano/branin.html>`_ and adding stochastic noise of
    the form

    :math:`u \cdot x + v \cdot y`

    where ``x`` and ``y`` are normally distributed with mean ``0.0`` and
    standard deviation ``1.0`` we get a loss function of the form

    :math:`(v - 5.1/(4 \cdot \pi^2) u^2 + 5/ \pi u - 6)^2 +\
    10 \cdot (1-1/(8 \cdot \pi)) \cdot \cos(u) + 10 + u \cdot x + v \cdot y`.

    Args:
      batch_size (int): Batch size to use.
      l2_reg (float): No L2-regularization (weight decay) is used in this
          test problem. Defaults to ``None`` and any input here is ignored.

    Attributes:
      dataset: The DeepOBS data set class for the two_d stochastic test problem.
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
    """

    def __init__(self, batch_size, l2_reg=None):
        """Create a new 2D Branin test problem instance.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): No L2-regularization (weight decay) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(two_d_branin, self).__init__(batch_size, l2_reg)

        if l2_reg is not None and l2_reg != 0.0:
            print(
                "WARNING: L2-Regularization is non-zero but no L2-regularization is used",
                "for this model.",
            )

    def set_up(self):
        """Sets up the stochastic two-dimensional Branin test problem.
        Using ``2.5`` and ``12.5`` as a starting point for the weights ``u``
        and ``v``.
        """
        self.dataset = two_d(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        x, y = self.dataset.batch

        # Set starting point
        starting_point = [2.5, 12.5]

        # Set model weights
        u = tf.compat.v1.get_variable(
            "weight", shape=(), initializer=tf.compat.v1.constant_initializer(starting_point[0]),
        )
        v = tf.compat.v1.get_variable(
            "bias", shape=(), initializer=tf.compat.v1.constant_initializer(starting_point[1]),
        )

        # Define some constants.
        a = 1.0
        b = 5.1 / (4.0 * np.pi ** 2)
        c = 5 / np.pi
        r = 6.0
        s = 10.0
        t = 1 / (8.0 * np.pi)

        self.losses = (
            a * (v - b * u ** 2 + c * u - r) ** 2
            + s * (1 - t) * tf.cos(u)
            + s
            + u * x
            + v * y
        )

        self.regularizer = tf.compat.v1.losses.get_regularization_loss()
