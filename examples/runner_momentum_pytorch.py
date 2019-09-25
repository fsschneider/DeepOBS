"""Example run script using StandardRunner."""

from torch.optim import SGD
from deepobs import pytorch as pt

optimizer_class = SGD
hyperparams = {"lr": {"type": float},
               "momentum": {"type": float, "default": 0.99},
               "nesterov": {"type": bool, "default": False}}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run(testproblem='quadratic_deep', hyperparams={'lr': 1e-2}, num_epochs=10)
