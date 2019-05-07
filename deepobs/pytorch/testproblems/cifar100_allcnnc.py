# -*- coding: utf-8 -*-
"""The all CNN-C architecture for CIFAR-100."""

import torch
from torch import nn
from .testproblems_modules import net_cifar100_allcnnc
from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem

class cifar100_allcnnc(TestProblem):
    def __init__(self, batch_size, weight_decay=0.0005):
        super(cifar100_allcnnc, self).__init__(batch_size, weight_decay)

    def set_up(self):
        """Set up the vanilla CNN test problem on Cifar-10."""
        self.data = cifar100(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.net = net_cifar100_allcnnc()
#        self._device = 'cpu'
#        torch.set_num_threads(12)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self._device)

    def get_regularization_loss(self):
        # iterate through all layers
        layer_norms = []
        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if 'bias' not in parameters_name:
                # L2 regularization
                layer_norms.append(parameters.pow(2).sum())

        regularization_loss = 0.5 * sum(layer_norms)

        return self._weight_decay * regularization_loss