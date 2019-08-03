# -*- coding: utf-8 -*-
"""A vanilla MLP architecture for MNIST."""

from torch import nn
from .testproblems_modules import net_mlp
from ..datasets.mnist import mnist
from .testproblem import UnregularizedTestproblem
import warnings

class mnist_mlp(UnregularizedTestproblem):
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
    weight_decay (float): No weight decay (L2-regularization) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

      Attributes:
        data: The DeepOBS data set class for Fashion-MNIST.
        loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss()
        net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_mlp).
        """
    def __init__(self, batch_size, weight_decay=None):
        """Create a new multi-layer perceptron test problem instance on \
        Fashion-MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(mnist_mlp, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            warnings.warn(
                "Weight decay is non-zero but no weight decay is used for this model.",
                RuntimeWarning
            )

    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = mnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_mlp(num_outputs=10)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()