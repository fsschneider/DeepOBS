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

The full documentation for the implementation with **TensorFlow** is available on readthedocs:
https://deepobs.readthedocs.io/

A very basic documentation for the implementation with **PyTorch** is available at:
https://deepobs_with_pytorch.readthedocs.io/

The paper describing DeepOBS has been accepted for ICLR 2019 and can be found
here:
https://openreview.net/forum?id=rJg6ssC5Y7

In this repository, the **PyTorch** implementation is developed. Additionally, DeepOBS will be refactored such that the **TensorFlow** code in this repo will change a lot. Please use the original
at https://github.com/fsschneider/DeepOBS if you want to use the **TensorFlow** code.

## Installation
	pip install -e git+https://github.com/abahde/DeepOBS.git@master#egg=DeepOBS

We tested the package with Python 3.6 and Torch version 1.0.1