============
Analyzer
============

DeepOBS uses the Analyzer class to get meaning full outputs from the results
created by the runners. This includes:

- Getting the best settings (e.g. best ``learning rate``) for all optimizers on every test problem of the benchmark.
- Plotting all performance metrices of the whole benchmark set.

The Analyzer bases its decisions on one of the four metrices:
- test accuracies (if available) 
- train accuraciesa(if available)
- test losses
- train losses

We distuingish three different modes of the analyzis:
- Best: The setting of the optimizer that led to the best perormance value on the whole learning curve.
- Final: The setting of the optimizer that led to the best performance after the last epoch.
- Most: The setting of the optimizer for which the most runs with different random seeds were executed. This is useful if you want to estimate the variance of the optimizer.

The Analyzer can include reference results. This means that the Analyzer will also plot and print the results for the references in the same output. This makes comparison to the own optimizer easier. For this, the argument ``reference_path`` needs to be set when creating the Analyzer instance. It is the path to the results folder of the references (e.g. ``<...>/baselines_deepobs``)