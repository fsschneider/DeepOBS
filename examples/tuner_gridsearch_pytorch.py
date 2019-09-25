from deepobs.tuner import GridSearch
from torch.optim import SGD
import numpy as np
from deepobs.pytorch.runners import StandardRunner

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float},
    "momentum": {"type": float},
    "nesterov": {"type": bool},
}

# The discrete values to construct a grid for.
grid = {
    "lr": np.logspace(-5, 2, 6),
    "momentum": [0.5, 0.7, 0.9],
    "nesterov": [False, True],
}

# Make sure to set the amount of ressources to the grid size. For grid search, this is just a sanity check.
tuner = GridSearch(
    optimizer_class,
    hyperparams,
    grid,
    runner=StandardRunner,
    ressources=6 * 3 * 2,
)

# Tune (i.e. evaluate every grid point) and rerun the best setting with 10 different seeds.
# tuner.tune('quadratic_deep', rerun_best_setting=True, num_epochs=2, output_dir='./grid_search')

# Optionally, generate commands for a parallelized execution
tuner.generate_commands_script(
    "quadratic_deep",
    run_script="../runner_momentum_pytorch.py",
    num_epochs=2,
    output_dir="./grid_search",
    generation_dir="./grid_search_commands",
)

