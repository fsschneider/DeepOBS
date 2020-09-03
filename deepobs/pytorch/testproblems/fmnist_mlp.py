# -*- coding: utf-8 -*-
"""A vanilla MLP architecture for Fashion-MNIST."""

import warnings

from torch import nn

from ..datasets.fmnist import fmnist
from .testproblem import UnregularizedTestproblem
from .testproblems_modules import net_mlp


class fmnist_mlp(UnregularizedTestproblem):
    """DeepOBS test problem class for a multi-layer perceptron neural network\
    on Fashion-MNIST.

  The network is build as follows:

    - Four fully-connected layers with ``1000``, ``500``, ``100`` and ``10``
      units per layer.
    - The first three layers use ReLU activation, and the last one a softmax
      activation.
    - The biases are initialized to ``0.0`` and the weight matrices with
      truncated normal (standard deviation of ``3e-2``)
    - The model uses a cross entropy loss.
    - No regularization is used.

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): No L2-Regularization (weight decay) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

      Attributes:
        data: The DeepOBS data set class for Fashion-MNIST.
        loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss()
        net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_mlp).
        """

    def __init__(self, batch_size, l2_reg=None):
        """Create a new multi-layer perceptron test problem instance on \
        Fashion-MNIST.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): No L2-Regularization (weight decay) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(fmnist_mlp, self).__init__(batch_size, l2_reg)

        if l2_reg is not None and l2_reg != 0.0:
            warnings.warn(
                "L2-Regularization is non-zero but no L2-regularization is used for this model.",
                RuntimeWarning,
            )

    def set_up(self):
        """Sets up the vanilla MLP test problem on Fashion-MNIST."""
        self.data = fmnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_mlp(num_outputs=10)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
