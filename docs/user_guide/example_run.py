"""Example run script using StandardRunner."""

import torch.optim as optim
import deepobs.pytorch as pyt

# specify the optimizer class
optimizer_class = optim.SGD

# and its hyperparameters
hyperparams = {'lr': 0.01,
               'momentum': 0.99}

# create the runner instance
runner = pyt.runners.StandardRunner(optimizer_class, hyperparams)

# run the optimizer on a testproblem
runner.run(testproblem ='mnist_2c2d',
           batch_size=128,
           num_epochs=10,
           output_dir='./results')

# possibly, run the same optimizer with a different setting again
#hyperparams = {'lr': 0.05,
#               'momentum': 0.9}
#runner = pyt.runners.StandardRunner(optimizer_class, hyperparams)
#runner.run(testproblem ='mnist_2c2d',
#                        batch_size=128,
#                        num_epochs=10,
#                        output_dir='./results')
