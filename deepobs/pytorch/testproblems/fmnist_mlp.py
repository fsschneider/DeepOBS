# -*- coding: utf-8 -*-
"""A vanilla MLP architecture for FMNIST."""

from torch import nn
from .testproblems_modules import net_mlp
from ..datasets.fmnist import fmnist
from .testproblem import TestProblem


class fmnist_mlp(TestProblem):


    def __init__(self, batch_size, weight_decay=None):
        """Create a new four layer MLP test problem instance on FMNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``0`` and any input here is ignored.
        """
        super(fmnist_mlp, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )


    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = fmnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.net = net_mlp(num_outputs=10)
        self.net.to(self._device)