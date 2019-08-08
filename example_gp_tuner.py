from deepobs.tuner import GP
from torch.optim import SGD
from sklearn.gaussian_process.kernels import Matern
from deepobs import config
from deepobs.pytorch.runners import StandardRunner

optimizer_class = SGD
hyperparams = {"lr": {"type": float},
               "momentum": {"type": float},
               "nesterov": {"type": bool}}

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


# The transformations of the search space. Momentum does not need a transformation.
transformations = {'lr': lr_transform,
                   'nesterov': nesterov_transform}

tuner = GP(optimizer_class, hyperparams, bounds, runner=StandardRunner, ressources=36, transformations=transformations)

# Tune with a Matern kernel and rerun the best setting with 10 different seeds.
tuner.tune('quadratic_deep', kernel=Matern(nu=2.5), rerun_best_setting=True, num_epochs=2, output_dir='./gp_tuner')
