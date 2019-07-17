Welcome to DeepOBS with PyTorch
===============================
This documentation covers the PyTorch API of DeepOBS.

**DeepOBS** is a benchmarking suite that drastically simplifies, automates and
improves the evaluation of deep learning optimizers.

It can evaluate the performance of new optimizers on a variety of
**real-world test problems** and automatically compare them with
**realistic baselines**.

DeepOBS automates several steps when benchmarking deep learning optimizers:

  - Downloading and preparing data sets.
  - Setting up test problems consisting of contemporary data sets and realistic
    deep learning architectures.
  - Running the optimizers on multiple test problems and logging relevant
    metrics.
  - Reporting and visualization the results of the optimizer benchmark.

The code for the current implementation working with **TensorFlow** and **PyTorch** can be found
on `GitHub`_.

=======
.. figure:: deepobs.jpg
    :scale: 40%


.. toctree::
  :maxdepth: 2
  :caption: User Guide

  user_guide/quick_start
  user_guide/tutorial

.. toctree::
  :maxdepth: 2
  :caption: API Reference

  api/analyzer

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. _GitHub: https://github.com/abahde/DeepOBS
