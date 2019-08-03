# -*- coding: utf-8 -*-
"""A vanilla MLP architecture for MNIST."""

from torch import nn
from .testproblems_modules import net_mnist_logreg
from ..datasets.mnist import mnist
from .testproblem import UnregularizedTestproblem
import warnings


class mnist_logreg(UnregularizedTestproblem):
    """DeepOBS test problem class for multinomial logistic regression on MNIST.

  No regularization is used and the weights and biases are initialized to ``0.0``.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): No weight decay (L2-regularization) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.
  """

    def __init__(self, batch_size, weight_decay=None):
        """Create a new multi-layer perceptron test problem instance on \
        Fashion-MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(mnist_logreg, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            warnings.warn(
                "Weight decay is non-zero but no weight decay is used for this model.",
                RuntimeWarning
            )

    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = mnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_mnist_logreg(num_outputs=10)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
