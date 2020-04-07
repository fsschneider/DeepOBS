import numpy as np
from torch.optim import SGD

from deepobs.config import get_small_test_set
from deepobs.pytorch.runners import StandardRunner
from deepobs.tuner import GridSearch

# define optimizer
optimizer_class = SGD
hyperparams = {"lr": {"type": float}}

### Grid Search ###
# The discrete values to construct a grid for.
grid = {"lr": np.logspace(-5, 2, 6)}

# init tuner class
tuner = GridSearch(
    optimizer_class, hyperparams, grid, runner=StandardRunner, ressources=6
)

# get the small test set and automatically tune on each of the contained test problems
small_testset = get_small_test_set()
tuner.tune_on_testset(
    small_testset, rerun_best_setting=True
)  # kwargs are parsed to the tune() method
