# -*- coding: utf-8 -*-
"""A simple 2D Noisy Beale Loss Function."""

import tensorflow as tf

from ..datasets.two_d import two_d
from .testproblem import TestProblem


class two_d_beale(TestProblem):
    r"""DeepOBS test problem class for a stochastic version of the\
    two-dimensional Beale function as the loss function.

    Using the deterministic `Beale function
    <https://www.sfu.ca/~ssurjano/beale.html>`_ and adding stochastic noise of
    the form

    :math:`u \cdot x + v \cdot y`

    where ``x`` and ``y`` are normally distributed with mean ``0.0`` and
    standard deviation ``1.0`` we get a loss function of the form

    :math:`((1.5 - u + u \cdot v)^2 + (2.25 - u + u \cdot v ^ 2) ^ 2 + (2.625 -\
    u + u \cdot v ^ 3) ^ 2) + u \cdot x + v \cdot y`.

    Args:
      batch_size (int): Batch size to use.
      weight_decay (float): No weight decay (L2-regularization) is used in this
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

    def __init__(self, batch_size, weight_decay=None):
        """Create a new 2D Beale Test Problem instance.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(two_d_beale, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model.",
            )

    def set_up(self):
        """Sets up the stochastic two-dimensional Beale test problem.
        Using ``-4.5`` and ``4.5`` as a starting point for the weights ``u``
        and ``v``.
        """
        self.dataset = two_d(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        x, y = self.dataset.batch

        # Set starting point
        starting_point = [-4.5, 4.5]

        # Set model weights
        u = tf.get_variable(
            "weight",
            shape=(),
            initializer=tf.constant_initializer(starting_point[0]),
        )
        v = tf.get_variable(
            "bias",
            shape=(),
            initializer=tf.constant_initializer(starting_point[1]),
        )

        self.losses = (
            (
                (1.5 - u + u * v) ** 2
                + (2.25 - u + u * v ** 2) ** 2
                + (2.625 - u + u * v ** 3) ** 2
            )
            + u * x
            + v * y
        )

        self.regularizer = tf.losses.get_regularization_loss()
