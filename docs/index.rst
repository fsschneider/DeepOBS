.. DeepOBS documentation master file, created by
   sphinx-quickstart on Wed Oct 10 10:29:24 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepOBS
===================

.. figure:: deepobs_banner.png

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
  - Reporting and visualizing the results of the optimizer benchmark.

.. figure:: deepobs.jpg
    :scale: 40%

The code for the current implementation working with **TensorFlow** can be found
on `GitHub`_.

We are actively working on a **PyTorch** version and will be releasing it in the
next months. In the meantime, PyTorch users can still use parts of DeepOBS such
as the data preprocessing scripts or the visualization features.

.. toctree::
  :maxdepth: 2
  :caption: User Guide

  user_guide/quick_start
  user_guide/overview
  user_guide/tutorial
  user_guide/suggested_protocol

.. toctree::
  :maxdepth: 2
  :caption: API Reference

  api/datasets
  api/testproblems
  api/runner
  api/analyzer
  api/scripts



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. _GitHub: https://github.com/fsschneider/DeepOBS
