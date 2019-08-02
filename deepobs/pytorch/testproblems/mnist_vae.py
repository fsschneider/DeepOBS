# -*- coding: utf-8 -*-
"""A VAE architecture for MNIST."""

import torch
from .testproblems_modules import net_vae
from ..datasets.mnist import mnist
from .testproblem import TestProblem
from .testproblems_utils import vae_loss_function_factory


class mnist_vae(TestProblem):
    """DeepOBS test problem class for a variational autoencoder (VAE) on \
    MNIST.

  The network has been adapted from the `here\
  <https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776>`_
  and consists of an encoder:

    - With three convolutional layers with each ``64`` filters.
    - Using a leaky ReLU activation function with :math:`\\alpha = 0.3`
    - Dropout layers after each convolutional layer with a rate of ``0.2``.

  and an decoder:

    - With two dense layers with ``24`` and ``49`` units and leaky ReLU activation.
    - With three deconvolutional layers with each ``64`` filters.
    - Dropout layers after the first two deconvolutional layer with a rate of ``0.2``.
    - A final dense layer with ``28 x 28`` units and sigmoid activation.

  No regularization is used.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): No weight decay (L2-regularization) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

  Attributes:
    data: The DeepOBS data set class for MNIST.
    loss_function: The loss function for this testproblem (vae_loss_function as defined in testproblem_utils)
    net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_vae).
  """

    def __init__(self, batch_size, weight_decay=None):
        """Create a new VAE test problem instance on MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(mnist_vae, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )

        self.loss_function = vae_loss_function_factory

    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.data = mnist(self._batch_size)
        self.net = net_vae(n_latent = 8)
        self.net.to(self._device)

    def get_batch_loss_and_accuracy(self, return_forward_func = False, reduction='mean',
                                    add_regularization_if_available=True):
        """Gets a new batch and calculates the loss and accuracy (if available)
        on that batch. This is a default implementation for image classification.
        Testproblems with different calculation routines (e.g. RNNs) overwrite this method accordingly.

        Args:
            return_forward_func (bool): If ``True``, the call also returns a function that calculates the loss on the current batch. Can be used if you need to access the forward path twice.
        Returns:
            float, float, (callable): loss and accuracy of the model on the current batch. If ``return_forward_func`` is ``True`` it also returns the function that calculates the loss on the current batch.
            """
        inputs, _ = self._get_next_batch()
        inputs = inputs.to(self._device)

        def _get_batch_loss_and_accuracy():
            # in evaluation phase is no gradient needed
            if self.phase in ["train_eval", "test", "valid"]:
                with torch.no_grad():
                    outputs, means, std_devs = self.net(inputs)
                    loss = self.loss_function(reduction=reduction)(outputs, inputs, means, std_devs)
            else:
                outputs, means, std_devs = self.net(inputs)
                loss = self.loss_function(reduction=reduction)(outputs, inputs, means, std_devs)

            accuracy = 0

            if add_regularization_if_available:
                regularizer_loss = self.get_regularization_loss()
            else:
                regularizer_loss = torch.tensor(0.0, device=torch.device(self._device))

            return loss + regularizer_loss, accuracy

        if return_forward_func:
            return _get_batch_loss_and_accuracy(), _get_batch_loss_and_accuracy
        else:
            return _get_batch_loss_and_accuracy()

