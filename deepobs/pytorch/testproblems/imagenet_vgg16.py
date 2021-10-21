# -*- coding: utf-8 -*-
"""VGG 16 architecture for ImageNet."""

from torch import nn

from ..datasets.imagenet import imagenet
from .testproblem import WeightRegularizedTestproblem
from .testproblems_modules import net_vgg


class imagenet_vgg16(WeightRegularizedTestproblem):
    """DeepOBS test problem class for the VGG 16 network on ImageNet.

    Details about the architecture can be found in the `original paper`_.
    VGG 16 consists of 16 weight layers, of mostly convolutions. The model uses
    cross-entroy loss. L2-Regularization is used on the weights (but not the biases)
    which defaults to ``5e-4``.

    .. _original paper: https://arxiv.org/abs/1409.1556

    Args:
      batch_size (int): Batch size to use.
      l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
          is used on the weights but not the biases.
          Defaults to ``5e-4``.

    Attributes:
      data: The DeepOBS data set class for ImageNet.
      loss_function: The loss function for this testproblem is
        torch.nn.CrossEntropyLoss()
      net: The DeepOBS subclass of torch.nn.Module that is trained for this
        tesproblem (net_vgg).
    """

    def __init__(self, batch_size, l2_reg=0.0005):
        """Create a new VGG 16 test problem instance on ImageNet.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(imagenet_vgg16, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the VGG 16 test problem on ImageNet."""
        self.data = imagenet(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_vgg(num_outputs=100, variant=16)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
