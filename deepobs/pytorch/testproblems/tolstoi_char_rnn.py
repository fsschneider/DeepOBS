# -*- coding: utf-8 -*-
"""A vanilla RNN architecture for Tolstoi."""
from torch import nn

from deepobs.pytorch.testproblems.testproblem import WeightRegularizedTestproblem
from .testproblems_modules import net_char_rnn
from ..datasets.tolstoi import tolstoi


class tolstoi_char_rnn(WeightRegularizedTestproblem):
    """DeepOBS test problem class for char_rnn network on Tolstoi.

    TODO: add some more details how the test problem works
    """

    # TODO check differences compared to tensorflow
    # - often the test on cuda fails: acc is greater than 1.0
    # - loss function:
    #   - tensorflow: mean across time, sum across batch
    #   - pytorch: mean across all

    def __init__(self, batch_size, l2_reg=0.0005):
        """Create a new char_rnn test problem instance on Tolstoi.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        print(f"batch_size={batch_size}")
        super(tolstoi_char_rnn, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the Char RNN test problem on Tolstoi."""
        self.data = tolstoi(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_char_rnn(hidden_dim=128, num_layers=2, seq_len=50, vocab_size=83)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
