from torch import nn

from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem
from .testproblems_modules import net_wrn


class cifar100_wrn164(TestProblem):
    """DeepOBS test problem class for the Wide Residual Network 16-4 architecture\
    for Cifar-100.

  Details about the architecture can be found in the `original paper`_.
  L2-Regularization is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  Training settings recommended in the `original paper`_:
  ``batch size = 128``, ``num_epochs = 160`` using the Momentum optimizer
  with :math:`\\mu = 0.9` and an initial learning rate of ``0.01`` with a decrease by
  ``0.1`` after ``80`` and ``120`` epochs.

  .. _original paper: https://arxiv.org/abs/1605.07146

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
        is used on the weights but not the biases.
        Defaults to ``5e-4``.
  """

    def __init__(self, batch_size, l2_reg=0.0005):
        """Create a new WRN 16-4 test problem instance on Cifar-100

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(cifar100_wrn164, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the Wide ResNet 16-4 test problem on Cifar-100."""
        self.data = cifar100(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_wrn(num_outputs=100, num_residual_blocks=2, widening_factor=4)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

    # TODO: Refactor, use WeightRegularizedTestproblem
    def get_regularization_groups(self):
        """Creates regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no, l2 = 0.0, self._l2_reg
        group_dict = {no: [], l2: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if ("weight" in parameters_name) and (
                ("dense" in parameters_name) or ("conv" in parameters_name)
            ):
                group_dict[l2].append(parameters)
            else:
                group_dict[no].append(parameters)
        return group_dict
