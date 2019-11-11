from deepobs.tuner import RandomSearch
from torch.optim import SGD
from deepobs.tuner.tuner_utils import log_uniform
from scipy.stats.distributions import uniform, binom
from deepobs import config
from deepobs.pytorch.runners import StandardRunner

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float},
    "momentum": {"type": float},
    "nesterov": {"type": bool},
}

# Define the distributions to sample from
distributions = {
    "lr": log_uniform(-5, 2),
    "momentum": uniform(0.5, 0.5),
    "nesterov": binom(1, 0.5),
}

# Allow 36 random evaluations.
tuner = RandomSearch(
    optimizer_class,
    hyperparams,
    distributions,
    runner=StandardRunner,
    ressources=36,
)

# Tune (i.e. evaluate 36 different random samples) and rerun the best setting with 10 different seeds.
tuner.tune(
    "quadratic_deep",
    rerun_best_setting=True,
    num_epochs=2,
    output_dir="./random_search",
)

# Optionally, generate commands for a parallelized execution
tuner.generate_commands_script(
    "quadratic_deep",
    run_script="../runner_momentum_pytorch.py",
    num_epochs=2,
    output_dir="./random_search",
    generation_dir="./random_search_commands",
)

