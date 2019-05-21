============
Quick Start
============

**DeepOBS** is a Python package to benchmark deep learning optimizers.
It currently supports TensorFlow and PyTorch.

Installation
============

You can install the latest DeepOBS with PyTorch implementation using `pip`:

.. code-block:: bash

   pip install -e git+https://github.com/abahde/DeepOBS.git@master#egg=DeepOBS

.. NOTE::
  The package requires the following packages:

  - argparse
  - numpy
  - pandas
  - matplotlib
  - matplotlib2tikz
  - torch (version 1.0.1)

Set-Up Data Sets
================

The PyTorch version of DeepOBS downloads image data sets via torchvision automatically if they are required by the testproblem.

Other data sets are not supported yet (e.g. text data for RNN character modelling).

You are now ready to run your optimizers on testproblems. 

We provide a :doc:`tutorial` for this.

Contributing to DeepOBS
=======================

The PyTorch version of Deepbs is under continuous development. Please report all bugs and remarks to `Aaron Bahde`_ (either as an issue on GitHub, via e-mail or in person)

.. _Aaron Bahde: https://github.com/abahde  
