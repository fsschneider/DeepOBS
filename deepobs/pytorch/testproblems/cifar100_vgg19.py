# -*- coding: utf-8 -*-
"""VGG 19 architecture for CIFAR-100."""
from torch import nn

from ..datasets.cifar100 import cifar100
from .testproblem import WeightRegularizedTestproblem
from .testproblems_modules import net_vgg


class cifar100_vgg19(WeightRegularizedTestproblem):
    """DeepOBS test problem class for the VGG 19 network on Cifar-100.

  The CIFAR-100 images are resized to ``224`` by ``224`` to fit the input
  dimension of the original VGG network, which was designed for ImageNet.

  Details about the architecture can be found in the `original paper`_.
  VGG 19 consists of 19 weight layers, of mostly convolutions. The model uses
  cross-entroy loss. L2-Regularization is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  .. _original paper: https://arxiv.org/abs/1409.1556

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
        is used on the weights but not the biases.
        Defaults to ``5e-4``.

  Attributes:
    data: The DeepOBS data set class for Cifar-100.
    loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss()
    net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_vgg).
  """

    def __init__(self, batch_size, l2_reg=0.0005):
        """Create a new VGG 19 test problem instance on Cifar-100.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(cifar100_vgg19, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the VGG 19 test problem on Cifar-100."""
        self.data = cifar100(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_vgg(num_outputs=100, variant=19)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
