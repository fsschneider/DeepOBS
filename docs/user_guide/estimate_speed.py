from torch.optim import Adam

from deepobs.analyzer.analyze import estimate_runtime, plot_results_table
from deepobs.pytorch.runners import StandardRunner

# plot the overview table which contains the speed measure for iterations
plot_results_table(
    "<path to your results>",
    conv_perf_file="<path to the convergence performance file of the baselines>",
)

# briefly run your optimizer against SGD to estimate wall-clock time overhead, here we use Adam as an example
estimate_runtime(
    framework="pytorch",
    runner_cls=StandardRunner,
    optimizer_cls=Adam,
    optimizer_hp={"lr": {"type": float}},
    optimizer_hyperparams={"lr": 0.1},
)
