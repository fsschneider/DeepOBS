# -*- coding: utf-8 -*-
"""The DCGAN architecture for F-MNIST."""
import warnings
import torch
from torch import nn


from ..datasets.fmnist import fmnist
from .testproblem import UnregularizedTestproblem
from .testproblems_modules import net_dcgan_g, net_dcgan_d
from .testproblems_utils import weights_init



class fmnist_dcgan(UnregularizedTestproblem):
    """DeepOBS test problem class for the Generative
    Adversarial Network DC architecture for Fashion-MNIST
    No regularization is used

    Training settings recommended in the `original paper`_:
  ``batch size = 128``, using the Adam optimizer
  with : ``beta1 = 0.5`` and an initial learning rate of ``0.0002``

  .. _original paper: https://arxiv.org/abs/1511.06434


    Args:
    batch_size (int): Batch size to use.
    l2_reg (float): No L2-Regularization (weight decay) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.


    Attributes:
    data: The DeepOBS data set class for Fashion-MNIST.
    loss_function: The loss function for this testproblem
    net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_dcgan_d)
    generator: The DeepOBS subclass of torch.nn.Module that is trained for this testproblem (net_dcgan_g)    """
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
        self.data = fmnist(self._batch_size, resize_images=True, train_eval_size=2048)
        self.loss_function = nn.BCELoss()
        self.generator = net_dcgan_g(num_channels=1)
        self.net = net_dcgan_d(num_channels=1)
        self.generator.to(self._device)
        self.net.to(self._device)
        self.generator.apply(weights_init)
        self.net.apply(weights_init)

