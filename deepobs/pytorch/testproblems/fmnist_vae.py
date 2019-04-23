# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for MNIST."""

import torch
from .testproblems_modules import net_vae
from ..datasets.fmnist import fmnist
from .testproblem import TestProblem


class fmnist_vae(TestProblem):
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
        super(fmnist_vae, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )

        def loss_function(outputs, targets, mean, std_dev):
            outputs_flat = outputs.view(-1, 28*28)
            targets_flat = targets.view(-1, 28*28)
            image_loss = torch.mean((outputs_flat - targets_flat).pow(2).sum(dim=1))
            latent_loss = -0.5 * torch.mean((1 + 2 * std_dev - mean.pow(2) - torch.exp(2*std_dev)).sum(dim=1))
            return image_loss + latent_loss

        self.loss_function=loss_function


    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = fmnist(self._batch_size)
        self.net = net_vae(n_latent = 8)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self._device)


    def get_batch_loss_and_accuracy(self):
        # Attention: loss is a tensor, accuracy a scalar
        # TODO in training phase the accuracy is calculated although not needed
        inputs, _ = self._get_next_batch()
        inputs = inputs.to(self._device)

        # in evaluation phase is no gradient needed
        if self.phase in ["train_eval", "test"]:
            with torch.no_grad():
                outputs, means, std_devs = self.net(inputs)
                loss = self.loss_function(outputs, inputs, means, std_devs)
        else:
            outputs, means, std_devs = self.net(inputs)
            loss = self.loss_function(outputs, inputs, means, std_devs)


        accuracy = 0
        return loss, accuracy

