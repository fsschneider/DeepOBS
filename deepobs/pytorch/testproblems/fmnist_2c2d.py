# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for Fashion MNIST."""

import torch
from torch import nn
from .testproblems_modules import net_mnist_2c2d
from ..datasets.fmnist import fmnist
from .testproblem import TestProblem


class fmnist_2c2d(TestProblem):
    """DeepOBS test problem class for a two convolutional and two dense layered\
    neural network on Fashion MNIST.
    """



    def __init__(self, batch_size, weight_decay=None):
        """Create a new 2c2d test problem instance on Fashion MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``0`` and any input here is ignored.
        """
        super(fmnist_2c2d, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )


    def set_up(self):
        """Sets up the vanilla CNN test problem on FMNIST."""
        self.data = fmnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.net = net_mnist_2c2d(num_outputs=10)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self._device)

    def get_batch_loss_and_accuracy(self):
        inputs, labels = self._get_next_batch()
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)
        correct = 0.0
        total = 0.0

        # in evaluation phase is no gradient needed
        if self.phase in ["train_eval", "test"]:
            with torch.no_grad():
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
        else:
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = correct/total
        return loss, accuracy