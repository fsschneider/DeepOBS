# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for MNIST."""

import torch
from torch import nn
from .testproblems_modules import net_mnist_2c2d
from ..datasets.mnist import mnist
from .testproblem import TestProblem


class mnist_2c2d(TestProblem):
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
    dataset: The DeepOBS data set class for MNIST.
    train_init_op: A tensorflow operation initializing the test problem for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the test problem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the test problem for
        evaluating on test data.
    losses: A tf.Tensor of shape (batch_size, ) containing the per-example loss
        values.
    regularizer: A scalar tf.Tensor containing a regularization term.
        Will always be ``0.0`` since no regularizer is used.
    accuracy: A scalar tf.Tensor containing the mini-batch mean accuracy.
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
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )


    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = mnist(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.net = net_mnist_2c2d(num_outputs=10)

    def get_batch_loss_and_accuracy(self):
        # Attention: loss is a tensor, accuracy a scalar
        # TODO in training phase the accuracy is calculated although not needed
        inputs, labels = self._get_next_batch()
        correct = 0.0
        total = 0.0

        # in evaluation phase is no gradient needed
        if self.phase in ["train_eval", "test"]:
            with torch.no_grad():
                outputs = self.net(inputs)
#                labels = labels.long()
                loss = self.loss_function(outputs, labels)
        else:
            outputs = self.net(inputs)
#            labels = labels.long()
            loss = self.loss_function(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = correct/total
        return loss, accuracy

