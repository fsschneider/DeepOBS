============
Quick Start
============

**DeepOBS** is a Python package to benchmark deep learning optimizers.
It supports TensorFlow and PyTorch.
We tested the package with Python 3.6, TensorFlow version 1.12 and Torch version 1.1.0.
Other versions of Python and TensorFlow (>= 1.4.0) might work, and we plan to
expand compatibility in the future.


Installation
============

You can install the latest DeepOBS with PyTorch implementation using `pip`:

.. code-block:: bash

   pip install -e git+https://github.com/abahde/DeepOBS.git@master#egg=DeepOBS

.. NOTE::
  Apart from Python 3.6 or higher, the package requires the following packages:

  - argparse
  - numpy
  - pandas
  - matplotlib
  - matplotlib2tikz
  - seaborn

  TensorFlow is not a required package to allow for both the CPU and GPU version. Make sure that one of those is installed. Additionally, you have to install torch/torchvision if you want to use the PyTorch framework.

.. HINT::
  We do not specify the exact version of the required package. However, if any
  problems occur while using DeepOBS, it might be a good idea to upgrade those
  packages to the newest release (especially matplotlib and numpy).

Set-Up Data Sets
================
**If you use TensorFlow**, you have to download the data sets for the test
problems. This can be done by simply running the
:doc:`../api/scripts/deepobs_prepare_data` script:

.. code-block:: bash

  deepobs_prepare_data.sh

Other data sets are not supported yet (e.g. text data for RNN character modelling).

You are now ready to run your optimizers on testproblems. 

We provide a :doc:`tutorial` for this.
=======
.. HINT::
  If you already have some of the data sets on your computer, you can only
  download the rest. If you have all data sets, you can skip this step, and
  always tell DeepOBS where the data sets are located. However, the DeepOBS
  package requires the data sets to be organized in a specific way.

**If you use PyTorch**, the data downloading will be handled automatically by torchvision.

You are now ready to run different optimizers on various test problems. We
provide a :doc:`tutorial` for this, as well as our
:doc:`suggested_protocol` for benchmarking deep learning optimizers.
>>>>>>> development

Contributing to DeepOBS
=======================

The PyTorch version of Deepbs is under continuous development. Please report all bugs and remarks to `Aaron Bahde`_ (either as an issue on GitHub, via e-mail or in person)

.. _Aaron Bahde: https://github.com/abahde  
