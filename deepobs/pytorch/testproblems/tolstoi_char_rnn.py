# -*- coding: utf-8 -*-
"""A vanilla RNN architecture for Tolstoi."""
from torch import nn

from deepobs.pytorch.testproblems.testproblem import WeightRegularizedTestproblem
from .testproblems_modules import net_char_rnn
from ..datasets.tolstoi import tolstoi


class tolstoi_char_rnn(WeightRegularizedTestproblem):
    """DeepOBS test problem class for a two-layer LSTM for character-level language
    modelling (Char RNN) on Tolstoi's War and Peace.

    Some network characteristics:

    - ``128`` hidden units per LSTM cell
    - sequence length ``50``
    - cell state is automatically stored in variables between subsequent steps
    - when the phase placeholder switches its value from one step to the next,
      the cell state is set to its zero value (meaning that we set to zero state
      after each round of evaluation, it is therefore important to set the
      evaluation interval such that we evaluate after a full epoch.)

    Working training parameters are:

    - batch size ``50``
    - ``200`` epochs
    - SGD with a learning rate of :math:`\\approx 0.1` works

    Args:
        batch_size (int): Batch size to use.
        l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
            is used on the weights but not the biases.
            Defaults to ``5e-4``.

    Attributes:
        data: The dataset used by the test problem (datasets.DataSet instance).
        loss_function: The loss function for this test problem.
        net: The torch module (the neural network) that is trained.
    """

    def __init__(self, batch_size, l2_reg=0.0005):
        """Create a new char_rnn test problem instance on Tolstoi.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(tolstoi_char_rnn, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the Char RNN test problem on Tolstoi."""
        self.data = tolstoi(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_char_rnn(hidden_dim=128, num_layers=2, seq_len=50, vocab_size=83)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()
