# DeepOBS - A Deep Learning Optimizer Benchmark Suite

![DeepOBS](docs/deepobs_banner.png "DeepOBS")

[![PyPI version](https://badge.fury.io/py/deepobs.svg)](https://badge.fury.io/py/deepobs)
[![Documentation Status](https://readthedocs.org/projects/deepobs/badge/?version=latest)](https://deepobs.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


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

![DeepOBS Output](docs/deepobs.jpg "DeepOBS_output")

This branch contains the beta of version 1.2.0 with **TensorFlow** and **PyTorch** support.
It is currently in a pre-release state.
Not all features are implemented and most notably we currently don't provide baselines for this version.

The full documentation of this beta version is available on readthedocs:
https://deepobs-with-pytorch.readthedocs.io/

The paper describing DeepOBS has been accepted for ICLR 2019 and can be found
here:
https://openreview.net/forum?id=rJg6ssC5Y7

**If you find any bugs in DeepOBS, or find it hard to use, please let us know.
We are always interested in feedback and ways to improve DeepOBS.**

## Installation

```pip install -e git+https://github.com/fsschneider/DeepOBS.git@version-1.2.0#egg=DeepOBS```

We tested the package with Python 3.6, TensorFlow version 1.12, Torch version 1.1.0 and Torchvision version 0.3.0.
Other versions might work, and we plan to expand compatibility in the future.

Further tutorials and a suggested protocol for benchmarking deep learning
optimizers can be found on https://deepobs-with-pytorch.readthedocs.io/
