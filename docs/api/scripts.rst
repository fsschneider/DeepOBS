============
Scripts
============

DeepOBS includes a few convenience scripts that can be run directly from the
command line

  - **Prepare Data**: Takes care of downloading and preprocessing all data sets
    for DeepOBS.
  - **Prepare Data**: Automatically downloads the baselines of DeepOBS.
  - **Plot Results**: Quickly plots the suggested outputs of a optimizer
    benchmark.
  - **Estimate Runtime**: Allows to estimate the runtime overhead of a new
    optimizer compared to SGD.

.. toctree::
  :maxdepth: 1
  :caption: Scripts

  scripts/deepobs_prepare_data
  scripts/deepobs_get_baselines
  scripts/deepobs_plot_results
  scripts/deepobs_estimate_runtime
