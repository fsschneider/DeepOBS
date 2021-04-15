"""Simple network architectures for DeepOBS in PyTorch."""

import numpy as np
import torch
from torch import nn

from deepobs.models.pytorch._utils import _truncated_normal_init

# TODO Beal, Branin, Rosenbrock


class QuadraticDeep(nn.Sequential):
    r"""Creates a network that corresponds to a quadratic problem.

    The loss function has the form:

    :math:`(\theta - x)^T * Q * (\theta - x)`

    with Hessian ``Q`` and "data" ``x`` coming from the quadratic data set, i.e.,
    zero-mean normal.
    The parameters are initialized to ``1.0``.
    """

    def __init__(self, hessian):
        """Build the network.

        Args:
            hessian (np.array): The matrix for the quadratic form.
        """
        super(QuadraticDeep, self).__init__()

        # for init
        if isinstance(hessian, (np.ndarray)):
            hessian = torch.from_numpy(hessian).to(torch.float32)
        dim = hessian.size(0)
        sqrt_hessian = self._compute_sqrt(hessian)

        self.add_module("shift", nn.Linear(dim, dim, bias=True))
        self.add_module("scale", nn.Linear(dim, dim, bias=False))

        # init
        self.shift.weight.data = -torch.eye(dim, dim)
        self.shift.weight.requires_grad = False
        nn.init.ones_(self.shift.bias)

        self.scale.weight.data = sqrt_hessian.t()
        self.scale.weight.requires_grad = False

    @staticmethod
    def _compute_sqrt(mat):
        return torch.cholesky(mat)


class LogReg(nn.Sequential):
    """Logistic Regression model."""

    def __init__(self, num_outputs):
        """Build the network.

        Args:
            num_outputs (int, optional): The numer of outputs (i.e. target classes).
                Defaults to ``10``.
        """
        super(LogReg, self).__init__()

        self.add_module("flatten", nn.Flatten())
        self.add_module("dense", nn.Linear(in_features=784, out_features=num_outputs))

        # init
        nn.init.constant_(self.dense.bias, 0.0)
        nn.init.constant_(self.dense.weight, 0.0)


class MLP(nn.Sequential):
    """Basic multi-layer perceptron.

    The network is build as follows:
    - Four fully-connected layers with ``1000``, ``500``,``100`` and ``num_outputs``
      units per layer, where ``num_outputs`` is the number of ouputs
      (i.e. class labels).
    - The first three layers use ReLU activation, and the last one a softmax
      activation.
    - The biases are initialized to ``0.0`` and the weight matrices with
      truncated normal (standard deviation of ``3e-2``).
    """

    def __init__(self, num_outputs=10):
        """Build the network.

        Args:
            num_outputs (int, optional): The numer of outputs (i.e. target classes).
                Defaults to ``10``.
        """
        super(MLP, self).__init__()

        self.add_module("flatten", nn.Flatten())
        self.add_module("dense1", nn.Linear(784, 1000))
        self.add_module("relu1", nn.ReLU())
        self.add_module("dense2", nn.Linear(1000, 500))
        self.add_module("relu2", nn.ReLU())
        self.add_module("dense3", nn.Linear(500, 100))
        self.add_module("relu3", nn.ReLU())
        self.add_module("dense4", nn.Linear(100, num_outputs))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=3e-2
                )
