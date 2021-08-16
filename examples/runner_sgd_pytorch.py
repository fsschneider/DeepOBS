"""StandardRunner: Default SGD."""

from torch.optim import SGD

from deepobs import pytorch as pt

optimizer_class = SGD
hyperparams = {"lr": {"type": float}}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run()
