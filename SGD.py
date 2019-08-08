from deepobs.pytorch.runners.runner import StandardRunner
from torch.optim.sgd import SGD
hyperparams = {'lr': {'type': float}, 'momentum': {'type': float}, 'nesterov': {'type': bool}}
runner = StandardRunner(SGD, hyperparams)
runner.run()