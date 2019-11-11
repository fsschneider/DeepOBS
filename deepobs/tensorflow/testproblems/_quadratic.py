# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:07:47 2018

@author: lballes
"""

import numpy as np
import tensorflow as tf

from ..datasets.quadratic import quadratic
from .testproblem import TestProblem


class _quadratic_base(TestProblem):
    r"""DeepOBS base class for a stochastic quadratic test problems creating loss\
    functions of the form

    :math:`0.5* (\theta - x)^T * Q * (\theta - x)`

    with Hessian ``Q`` and "data" ``x`` coming from the quadratic data set, i.e.,
    zero-mean normal.

    Args:
        batch_size (int): Batch size to use.
        weight_decay (float): No weight decay (L2-regularization) is used in this
            test problem. Defaults to ``None`` and any input here is ignored.
        hessian (np.array): Hessian of the quadratic problem.
            Defaults to the ``100`` dimensional identity.

    Attributes:
      dataset: The DeepOBS data set class for the quadratic test problem.
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

    def __init__(self, batch_size, weight_decay=None, hessian=np.eye(100)):
        """Create a new quadratic test problem instance.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): No weight decay (L2-regularization) is used in this
                test problem. Defaults to ``None`` and any input here is ignored.
            hessian (np.array): Hessian of the quadratic problem.
                Defaults to the ``100`` dimensional identity.
        """
        super(_quadratic_base, self).__init__(batch_size, weight_decay)
        self._hessian = hessian
        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model.",
            )

    def set_up(self):
        """Sets up the stochastic quadratic test problem. The parameter ``Theta``
        will be initialized to (a vector of) ``1.0``.
        """
        self.dataset = quadratic(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        x = self.dataset.batch
        hessian = tf.convert_to_tensor(self._hessian, dtype=tf.float32)
        theta = tf.get_variable(
            "theta",
            shape=(1, hessian.shape[0]),
            initializer=tf.constant_initializer(1.0),
        )

        self.losses = tf.linalg.tensor_diag_part(
            0.5
            * tf.matmul(
                tf.subtract(theta, x),
                tf.matmul(hessian, tf.transpose(tf.subtract(theta, x))),
            )
        )
        self.regularizer = tf.losses.get_regularization_loss()
