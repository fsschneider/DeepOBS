# -*- coding: utf-8 -*-
"""A VAE architecture for MNIST."""

import torch
from .testproblems_modules import net_vae
from ..datasets.mnist import mnist
from .testproblem import UnregularizedTestproblem
from .testproblems_utils import vae_loss_function_factory


class mnist_vae(UnregularizedTestproblem):
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
        self.regularization_groups = self.get_regularization_groups()

    def get_batch_loss_and_accuracy_func(self,
                                    reduction='mean',
                                    add_regularization_if_available=True):
        """Get new batch and create forward function that calculates loss and accuracy (if available)
        on that batch.

        Args:
            reduction (str): The reduction that is used for returning the loss. Can be 'mean', 'sum' or 'none' in which \
            case each indivual loss in the mini-batch is returned as a tensor.
            add_regularization_if_available (bool): If true, regularization is added to the loss.
        Returns:
            callable:  The function that calculates the loss/accuracy on the current batch.
        """

        inputs, _ = self._get_next_batch()
        inputs = inputs.to(self._device)

        def forward_func():
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

        return forward_func
