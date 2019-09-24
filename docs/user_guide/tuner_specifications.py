from deepobs.tuner import GridSearch, RandomSearch, GP
from torch.optim import SGD
import numpy as np
from deepobs.pytorch.runners import StandardRunner
from deepobs.tuner.tuner_utils import log_uniform
from scipy.stats.distributions import uniform, binom

# define optimizer
optimizer_class = SGD
hyperparams = {"lr": {"type": float},
               "momentum": {"type": float},
               "nesterov": {"type": bool}}

### Grid Search ###
# The discrete values to construct a grid for.
grid = {'lr': np.logspace(-5, 2, 6),
        'momentum': [0.5, 0.7, 0.9],
        'nesterov': [False, True]}

# Make sure to set the amount of resources to the grid size. For grid search, this is just a sanity check.
tuner = GridSearch(optimizer_class, hyperparams, grid, runner=StandardRunner, ressources=6*3*2)

### Random Search ###
# Define the distributions to sample from
distributions = {'lr': log_uniform(-5, 2),
        'momentum': uniform(0.5, 0.5),
        'nesterov': binom(1, 0.5)}

# Allow 36 random evaluations.
tuner = RandomSearch(optimizer_class, hyperparams, distributions, runner=StandardRunner, ressources=36)

### Bayesian Optimization ###
# The bounds for the suggestions
bounds = {'lr': (-5, 2),
        'momentum': (0.5, 1),
        'nesterov': (0, 1)}


# Corresponds to rescaling the kernel in log space.
def lr_transform(lr):
    return 10**lr


# Nesterov is discrete but will be suggested continious.
def nesterov_transform(nesterov):
    return bool(round(nesterov))


# The transformations of the search space. The momentum parameter does not need a transformation.
transformations = {'lr': lr_transform,
                   'nesterov': nesterov_transform}

tuner = GP(optimizer_class, hyperparams, bounds, runner=StandardRunner, ressources=36, transformations=transformations)