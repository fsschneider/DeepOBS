# -*- coding: utf-8 -*-
"""VGG 16 architecture for CIFAR-100."""
from torch import nn

from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem
from .testproblems_modules import net_vgg

class cifar100_vgg16(TestProblem):
    """DeepOBS test problem class for the VGG 16 network on Cifar-100.

  The CIFAR-100 images are resized to ``224`` by ``224`` to fit the input
  dimension of the original VGG network, which was designed for ImageNet.

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
    data: The DeepOBS data set class for Cifar-100.
    loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss()
    net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_vgg).

  """


    def __init__(self, batch_size, l2_reg=0.0005):
        """Create a new VGG 16 test problem instance on Cifar-100.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(cifar100_vgg16, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the VGG 16 test problem on Cifar-100."""
        self.data = cifar100(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_vgg(num_outputs=100, variant=16)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

    def get_regularization_groups(self):
        """Creates regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no, l2 = 0.0, self._l2_reg
        group_dict = {no: [], l2: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if "bias" not in parameters_name:
                group_dict[l2].append(parameters)
            else:
                group_dict[no].append(parameters)
        return group_dict

