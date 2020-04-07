from deepobs.analyzer.analyze import (plot_hyperparameter_sensitivity,
                                      plot_optimizer_performance)

# plot your optimizer against baselines
plot_optimizer_performance(
    "/<path to your results folder>/<test problem>/<your optimizer>",
    reference_path="<path to the baselines>/<test problem>/SGD",
)

# plot the hyperparameter sensitivity (here we use the learning rate sensitivity of the SGD baseline)
plot_hyperparameter_sensitivity(
    "<path to the baselines>/<test problem>/SGD",
    hyperparam="lr",
    xscale="log",
    plot_std=True,
)
