============
Analyzer
============

DeepOBS uses the analyzer class to get meaning full outputs from the results
created by the runners. This includes:

- Getting the best settings (e.g. best ``learning rate``) for an optimizer on a specific test problem.
- Plotting the ``learning_rate`` sensitivity for multiple optimizers on a test problem.
- Plotting all performance metrics of the whole benchmark set.
- Returning the overall performance table for multiple optimizers.

The analyzer can return those outputs as matplotlib plots or ``.tex`` files for
direct inclusion in academic publications.

DeepOBS also includes a convenience script using this analyzer class for these
most used cases, see  :doc:`./scripts/deepobs_plot_results`

.. toctree::
  :maxdepth: 2
  :caption: Analyzer

  analyzer/analyzer
  analyzer/testproblemanalyzer
  analyzer/optimizeranalyzer
  analyzer/settinganalyzer
  analyzer/aggregaterun
