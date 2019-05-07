# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for MNIST."""

import torch
from torch import nn
from .testproblems_modules import net_mnist_2c2d
from ..datasets.mnist import mnist
from .testproblem import TestProblem


class mnist_2c2d(TestProblem):


    def __init__(self, batch_size, weight_decay=None):
        """Create a new 2c2d test problem instance on MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``0`` and any input here is ignored.
        """
        super(mnist_2c2d, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )


    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = mnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.net = net_mnist_2c2d(num_outputs=10)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self._device)