============
Analyzer
============

DeepOBS uses the analyzer class to get meaning full outputs from the results
created by the runners. This includes:

- Getting the best settings (e.g. best ``learning rate``) for an optimizer on a specific test problem.
- Plotting the hyperparameter (e.g. ``learning_rate``) sensitivity for multiple optimizers on a test problem.
- Plotting all performance metrics of the whole benchmark set.
- Returning the overall performance table for multiple optimizers.

The analyzer can return those outputs as matplotlib plots or ``.tex`` files for
direct inclusion in academic publications.


.. toctree::
  :maxdepth: 2
  :caption: Analyzer

  analyzer/analyze
