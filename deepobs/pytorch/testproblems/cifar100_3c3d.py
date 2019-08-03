# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for CIFAR-100."""

import torch
from torch import nn
from .testproblems_modules import net_cifar10_3c3d
from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem


class cifar100_3c3d(TestProblem):
    """DeepOBS test problem class for a three convolutional and three dense \
    layered neural network on Cifar-100.

  The network consists of

    - thre conv layers with ReLUs, each followed by max-pooling
    - two fully-connected layers with ``512`` and ``256`` units and ReLU activation
    - 100-unit output layer with softmax
    - cross-entropy loss
    - L2 regularization on the weights (but not the biases) with a default
      factor of 0.002

  The weight matrices are initialized using Xavier initialization and the biases
  are initialized to ``0.0``.

  Args:
      batch_size (int): Batch size to use.
      weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
          is used on the weights but not the biases. Defaults to ``0.002``.

  Attributes:
    data: The DeepOBS data set class for Cifar-100.
    loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss().
    net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_cifar10_3c3d with 100 outputs).

  Methods:
      get_regularization_loss: Returns the current regularization loss of the network state.
  """
    def __init__(self, batch_size, weight_decay=0.002):
        super(cifar100_3c3d, self).__init__(batch_size, weight_decay)

    def set_up(self):
        """Set up the vanilla CNN test problem on Cifar-100."""
        self.data = cifar100(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_cifar10_3c3d(num_outputs=100)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

    def get_regularization_groups(self):
        """Creates regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no, l2 = 0.0, self._weight_decay
        group_dict = {no: [], l2: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if 'bias' not in parameters_name:
                group_dict[l2].append(parameters)
            else:
                group_dict[no].append(parameters)
        return group_dict
