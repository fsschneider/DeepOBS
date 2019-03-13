============
Plot Results
============

A convenience script to extract useful information out of the results create by
the runners.

This script can return one or all of the below information:

  - Get best run: Returns the best hyperparameter setting for each optimizer in
    each test problem.
  - Plot learning rate sensitivity: Creates a plot for each test problem showing
    the relative performance of each optimizer against the learning rate to get
    a sense of how difficult the tuning process was.
  - Plot performance: Creates a plot for the ``small`` and ``large`` benchmark
    set, plotting (if available) all four performance metric (``losses`` and
    ``accuracies`` for both the test and the train data set) for each optimizer.
  - Plot table: Creates the overall performance table for the  ``small`` and
    ``large`` benchmark set including metrics for the performance, speed and
    tuneability of each optimizer on each test problem.

If the path to the baseline folder is given, this script will also plot the
performances of `SGD`, `Momentum` and `Adam`.

**Usage:**

.. argparse::
   :filename: ../deepobs/scripts/deepobs_plot_results.py
   :func: parse_args
   :prog: deepobs_plot_results.py
