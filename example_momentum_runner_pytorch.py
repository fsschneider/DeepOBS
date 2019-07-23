"""Example run script using StandardRunner."""

from torch.optim import SGD
from deepobs import pytorch as pt

optimizer_class = SGD
hyperparams = {"learning_rate": {"type": float},
               "momentum": {"type": float, "default": 0.99},
               "use_nesterov": {"type": bool, "default": False}}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run(testproblem='quadratic_deep', hyperparams={'lr': 0.1}, num_epochs=10)
