# -*- coding: utf-8 -*-
"""A logistic regression problem for Fashion MNIST."""

import warnings

from torch import nn

from ..datasets.fmnist import fmnist
from .testproblem import UnregularizedTestproblem
from .testproblems_modules import net_mnist_logreg


class fmnist_logreg(UnregularizedTestproblem):
    """DeepOBS test problem class for multinomial logistic regression on FMNIST.

  No regularization is used and the weights and biases are initialized to ``0.0``.

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): No L2-Regularization (weight decay) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.
  """

    def __init__(self, batch_size, l2_reg=None):
        """Create a new multi-layer perceptron test problem instance on \
        FMNIST.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): No L2-Regularization (weight decay) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(fmnist_logreg, self).__init__(batch_size, l2_reg)

        if l2_reg is not None:
            warnings.warn(
                "L2-Regularization is non-zero but no L2-regularization is used for this model.",
                RuntimeWarning,
            )

    def set_up(self):
        """Sets up the vanilla CNN test problem on FMNIST."""
        self.data = fmnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_mnist_logreg(num_outputs=10)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
