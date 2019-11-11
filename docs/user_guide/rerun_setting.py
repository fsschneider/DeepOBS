from deepobs.tuner import GridSearch
from torch.optim import SGD
import numpy as np
from deepobs.pytorch.runners import StandardRunner

# define optimizer
optimizer_class = SGD
hyperparams = {"lr": {"type": float}}

### Grid Search ###
# The discrete values to construct a grid for.
grid = {'lr': np.logspace(-5, 2, 6)}

# init tuner class
tuner = GridSearch(optimizer_class, hyperparams, grid, runner=StandardRunner, ressources=6)

# tune on quadratic test problem and automatically rerun the best instance with 10 different seeds.
tuner.tune('quadratic_deep', rerun_best_setting=True)