# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for MNIST."""

import warnings
from torch import nn
from .testproblems_modules import net_mnist_2c2d
from ..datasets.mnist import mnist
from .testproblem import UnregularizedTestproblem


class mnist_2c2d(UnregularizedTestproblem):
    """DeepOBS test problem class for a two convolutional and two dense layered\
    neural network on MNIST.

  The network has been adapted from the `TensorFlow tutorial\
  <https://www.tensorflow.org/tutorials/estimators/cnn>`_ and consists of

    - two conv layers with ReLUs, each followed by max-pooling
    - one fully-connected layers with ReLUs
    - 10-unit output layer with softmax
    - cross-entropy loss
    - No regularization

  The weight matrices are initialized with truncated normal (standard deviation
  of ``0.05``) and the biases are initialized to ``0.05``.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): No weight decay (L2-regularization) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

   Attributes:
    data: The DeepOBS data set class for MNIST.
    loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss().
    net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_mnist_2c2d).
  """

    def __init__(self, batch_size, weight_decay=None):
        """Create a new 2c2d test problem instance on MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``0`` and any input here is ignored.
        """
        super(mnist_2c2d, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            warnings.warn(
                "Weight decay is non-zero but no weight decay is used for this model.",
                RuntimeWarning
            )

    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = mnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_mnist_2c2d(num_outputs=10)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()