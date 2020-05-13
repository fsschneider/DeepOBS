# -*- coding: utf-8 -*-
"""The DCGAN architecture for F-MNIST."""
import warnings
import torch
from torch import nn

from ..datasets.fmnist import fmnist
from .testproblem import UnregularizedTestproblem
from .testproblems_modules import dcgan_g, dcgan_d
from .testproblems_utils import weights_init

class fmnist_dcgan(UnregularizedTestproblem):
    """DeepOBS test problem class for the Generative
    Adversarial Network DC architecture for Fashion-MNIST
    No regularization is used

    Args:
    batch_size (int): Batch size to use.
    l2_reg (float): No L2-Regularization (weight decay) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.


    Attributes:
    data: The DeepOBS data set class for Fashion-MNIST.
    loss_function: The loss function for this testproblem
    net:
    """
    def __init__(self, batch_size, l2_reg=None):
        """Create a new DCGAN test problem instance on Fashion-MNIST

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): No L2-Regularization (weight decay) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(fmnist_dcgan, self).__init__(batch_size,l2_reg)
        if l2_reg is not None:
            warnings.warn(
                "L2-Regularization is non-zero but no L2-regularization is used for this model.",
                RuntimeWarning,
            )

    def set_up(self):
        """Set up the DCGAN test problem on F-MNIST"""
        self.data = fmnist(self._batch_size, resize_images=True, train_eval_size=1)
        self.loss_function = nn.BCELoss()
        self.generator = dcgan_g(num_channels=1)
        self.net = dcgan_d(num_channels=1)
        self.generator.to(self._device)
        self.net.to(self._device)
        self.generator.apply(weights_init)
        self.net.apply(weights_init)

