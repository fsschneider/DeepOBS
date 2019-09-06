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

The code for the current implementation working with **TensorFlow** can be found
on [Github](https://github.com/fsschneider/DeepOBS).

The full documentation is available on readthedocs:
https://deepobs.readthedocs.io/

The paper describing DeepOBS has been accepted for ICLR 2019 and can be found
here:
https://openreview.net/forum?id=rJg6ssC5Y7

We are currently working on a new and improved version of DeepOBS.
It will support **PyTorch** as well as TensorFlow, has an easier interface, and
many bugs ironed out. It will be released in a few weeks, until then you can
check out the current working beta version of it over at https://github.com/abahde/DeepOBS.

**If you find any bugs in DeepOBS, or find it hard to use, please let us know.
We are always interested in feedback and ways to improve DeepOBS.**

## Installation

	pip install deepobs

We tested the package with Python 3.6 and TensorFlow version 1.12. Other
versions of Python and TensorFlow (>= 1.4.0) might work, and we plan to expand
compatibility in the future.

Further tutorials and a suggested protocol for benchmarking deep learning
optimizers can be found on https://deepobs.readthedocs.io/
