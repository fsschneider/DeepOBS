==================
Download Baselines
==================

A convenience script to download all baselines for DeepOBS.

.. NOTE::
  The download is currently around ``470`` MB large, so it might take a while,
  depending on your internet connection.

The baselines are currently for the three most popular deep learning optimizers,
``SGD``, ``Momentum`` and ``Adam``. The files include the ``JSON`` results for
both the hyperparameter tuning phase (``36`` runs with different learning rates)
as well as the final results with the best performing setting (``10`` runs with
different random seeds and the same hyperparameter setting).

They can be used together with the plotting module or script to automatically
compare the results of new optimizers, without having to run those baselines
again.

**Usage:**

  .. code:: python

    usage: deepobs_get_baselines.sh [--data_dir=DIR]

Named Arguments
===============

+---------------+----------------------------------------------------------------------------+
| -d --data_dir | Path where the baselines should be saved. Defaults to "baselines_deepobs". |
+---------------+----------------------------------------------------------------------------+
