import torch
from torch import nn
from .testproblems_modules import net_wrn
from ..datasets.svhn import svhn
from .testproblem import TestProblem


class svhn_wrn164(TestProblem):
    """DeepOBS test problem class for the Wide Residual Network 16-4 architecture\
    for SVHN.

  Details about the architecture can be found in the `original paper`_.
  A weight decay is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  Training settings recommenden in the `original paper`_:
  ``batch size = 128``, ``num_epochs = 160`` using the Momentum optimizer
  with :math:`\\mu = 0.9` and an initial learning rate of ``0.01`` with a decrease by
  ``0.1`` after ``80`` and ``120`` epochs.

  .. _original paper: https://arxiv.org/abs/1605.07146

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
        is used on the weights but not the biases.
        Defaults to ``5e-4``.
  """

    def __init__(self, batch_size, weight_decay=0.0005):
        """Create a new WRN 16-4 test problem instance on SVHN.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(svhn_wrn164, self).__init__(batch_size, weight_decay)

    def set_up(self):
        """Set up the Wide ResNet 16-4 test problem on SVHN."""
        self.data = svhn(self._batch_size, data_augmentation=True)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_wrn(num_outputs=10, num_residual_blocks=2, widening_factor=4)
        self.net.to(self._device)

    def get_regularization_loss(self):
        # iterate through all layers
        layer_norms = []
        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if ('weight' in parameters_name) and (('dense' in parameters_name) or ('conv' in parameters_name)):
                # L2 regularization
                layer_norms.append(parameters.pow(2).sum())

        regularization_loss = 0.5 * sum(layer_norms)

        return self._weight_decay * regularization_loss