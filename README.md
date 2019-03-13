# DeepOBS - A Deep Learning Optimizer Benchmark Suite

![DeepOBS](docs/deepobs_banner.png "DeepOBS")

[![Documentation Status](https://readthedocs.org/projects/deepobs/badge/?version=latest)](https://deepobs.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/fsschneider/deepobs.svg?branch=master)](https://travis-ci.com/username/projectname)
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

We are actively working on a **PyTorch** version and will be releasing it in the
next months. In the meantime, PyTorch users can still use parts of DeepOBS such
as the data preprocessing scripts or the visualization features.


## Installation

	pip install deepobs

We tested the package with Python 3.6 and TensorFlow version 1.12. Other
versions of Python and TensorFlow (>= 1.4.0) might work, and we plan to expand
compatibility in the future.

Further tutorials and a suggested protocol for benchmarking deep learning
optimizers can be found on https://deepobs.readthedocs.io/
