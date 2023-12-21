# DeepOBS - A Deep Learning Optimizer Benchmark Suite

----

> This package is no longer maintained.
> It is superseded by the [AlgoPerf benchmark suite](https://github.com/mlcommons/algorithmic-efficiency)

-----

![DeepOBS](docs/deepobs_banner.png "DeepOBS")

[![PyPI version](https://badge.fury.io/py/deepobs.svg)](https://badge.fury.io/py/deepobs)
[![Documentation Status](https://readthedocs.org/projects/deepobs/badge/?version=stable)](https://deepobs.readthedocs.io/en/stable/)
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
  - Reporting and visualizing the results of the optimizer benchmark.

![DeepOBS Output](docs/deepobs.jpg "DeepOBS_output")

The code for the current implementation working with **TensorFlow** can be found
on [Github](https://github.com/fsschneider/DeepOBS).
A PyTorch version is currently developed and can be accessed via the pre-release or the develop branch (see News section below).

The full documentation is available on readthedocs:
https://deepobs.readthedocs.io/

The paper describing DeepOBS has been accepted for ICLR 2019 and can be found
here:
https://openreview.net/forum?id=rJg6ssC5Y7

**If you find any bugs in DeepOBS, or find it hard to use, please let us know.
We are always interested in feedback and ways to improve DeepOBS.**

## News

We are currently working on a new and improved version of DeepOBS, version 1.2.0.
It will support **PyTorch** in addition to TensorFlow, has an easier interface, and
many bugs ironed out. You can find the latest version of it in [this branch](https://github.com/fsschneider/DeepOBS/tree/develop).

A [pre-release](https://github.com/fsschneider/DeepOBS/releases/tag/v1.2.0-beta0) is available now. 
The full release is expected in a few weeks.

Many thanks to [Aaron Bahde](https://github.com/abahde) for spearheading the development of DeepOBS 1.2.0.

## Installation

	pip install deepobs

We tested the package with Python 3.6 and TensorFlow version 1.12. Other
versions of Python and TensorFlow (>= 1.4.0) might work, and we plan to expand
compatibility in the future.

If you want to create a local and modifiable version of DeepOBS, you can do this directly from this repo via

	pip install -e git+https://github.com/fsschneider/DeepOBS.git#egg=DeepOBS

for the stable version, or 

	pip install -e git+https://github.com/fsschneider/DeepOBS.git@develop#egg=DeepOBS

for the latest development version.


Further tutorials and a suggested protocol for benchmarking deep learning
optimizers can be found on https://deepobs.readthedocs.io/
