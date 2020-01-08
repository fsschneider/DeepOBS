<p align="center"><img src="https://raw.githubusercontent.com/fsschneider/DeepOBS/master/docs/deepobs_banner.png" /></p>

# DeepOBS - A Deep Learning Optimizer Benchmark Suite.

[**Install Guide**](#installation)
| [**Documentation**](https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/)
| [**Examples**](https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/user_guide/tutorial.html)
| [**Paper**](https://openreview.net/forum?id=rJg6ssC5Y7)
| [**Leaderboard**](https://deepobs.github.io/#Leaderboard)
| [**Baselines**](https://github.com/fsschneider/DeepOBS_Baselines)

[![License: MIT](https://img.shields.io/github/license/fsschneider/deepobs?style=flat-square)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deepobs?style=flat-square)
[![PyPI version](https://img.shields.io/pypi/v/deepobs.svg?style=flat-square)](https://pypi.org/project/deepobs)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deepobs?style=flat-square)
![Codacy branch grade](https://img.shields.io/codacy/grade/0b6cb61af02745af8ed9126c7d0779e6/develop?logo=Codacy&style=flat-square)
[![Documentation Status](https://readthedocs.org/projects/deepobs/badge/?version=v1.2.0-beta0_a&style=flat-square)](https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/?badge=v1.2.0-beta0_a)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

--------------------------------------------------------------------------------

> ⚠️ **This branch contains the beta of version 1.2.0**
>
> It contains the latest changes planned for the release of DeepOBS 1.2.0, including support for **PyTorch**. Not all features are implemented and most notably we currently don't provide baselines for this version. We continuously make changes to this version, so things can break if you update. If you want a more stable preview, check out our pre-releases.

## 📇 Table of Contents

- [DeepOBS - A Deep Learning Optimizer Benchmark Suite.](#deepobs---a-deep-learning-optimizer-benchmark-suite)
  - [📇 Table of Contents](#%f0%9f%93%87-table-of-contents)
  - [📦 Introduction](#%f0%9f%93%a6-introduction)
  - [🆕 News](#%f0%9f%86%95-news)
  - [💻 Getting Started](#%f0%9f%92%bb-getting-started)
    - [Installation](#installation)
    - [Quick Start Guide](#quick-start-guide)
      - [Download Data Sets](#download-data-sets)
      - [Run your Optimizer](#run-your-optimizer)
      - [Further Steps](#further-steps)
  - [🏅 Leaderboard & Baselines](#%f0%9f%8f%85-leaderboard--baselines)
  - [👨‍👨‍👧‍👦 Contributors](#%f0%9f%91%a8%e2%80%8d%f0%9f%91%a8%e2%80%8d%f0%9f%91%a7%e2%80%8d%f0%9f%91%a6-contributors)
  - [📝 Citation](#%f0%9f%93%9d-citation)

## 📦 Introduction

**DeepOBS** is a python framework for automatically benchmarking deep learning optimizers. Its goal is to drastically simplify, automate and improve the evaluation of deep learning optimizers.

It can:

- evaluate the performance of new optimizers on more than **25 real-world test problems**, such as training Residual Networks for image classification or LSTMs for character prediction.
- automatically compare the results with **realistic baselines** (without having to run them again!)

DeepOBS automates several steps when benchmarking deep learning optimizers:

- Downloading and preparing data sets.
- Setting up test problems consisting of contemporary data sets and realistic deep learning architectures.
- Running the optimizers on multiple test problems and logging relevant metrics.
- Comparing your results to the newest baseline results of other optimizers.
- Reporting and visualizing the results of the optimizer benchmark as ready-to-include ``.tex`` files.

![DeepOBS Output](https://raw.githubusercontent.com/fsschneider/DeepOBS/develop/docs/deepobs.jpg "DeepOBS_output")

The code for the current implementation working with **TensorFlow** can be found on [Github](https://github.com/fsschneider/DeepOBS).
A PyTorch version is currently developed (see News section below).

The documentation of the beta version is available on [readthedocs](https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/).

The [paper describing DeepOBS](https://openreview.net/forum?id=rJg6ssC5Y7) has been accepted for ICLR 2019.

**If you find any bugs in DeepOBS, or find it hard to use, please let us know. We are always interested in feedback and ways to improve DeepOBS.**

## 🆕 News

We are currently working on a new and improved version of DeepOBS, version 1.2.0.
It will support **PyTorch** in addition to TensorFlow, has an easier interface, and
many bugs ironed out. You can find the latest version of it in [this branch](https://github.com/fsschneider/DeepOBS/tree/v1.2.0-beta0).

A pre-release is currently available and a full release will be available in the coming weeks.

Many thanks to [Aaron Bahde](https://github.com/abahde) for spearheading the development of DeepOBS 1.2.0.

## 💻 Getting Started

### Installation

    pip install deepobs

We tested the package with Python 3.6, TensorFlow version 1.12, Torch version 1.1.0 and Torchvision version 0.3.0.
Other versions of Python and TensorFlow (>= 1.4.0) might work, and we plan to expand compatibility in the future.

If you want to create a local and modifiable version of DeepOBS, you can do this directly from this repo via

    pip install -e git+https://github.com/fsschneider/DeepOBS.git#egg=DeepOBS

for the latest stable version, or

    pip install -e git+https://github.com/fsschneider/DeepOBS.git@v1.2.0-beta0#egg=DeepOBS

to get the preview of DeepOBS 1.2.0.

### Quick Start Guide

#### Download Data Sets

If you use TensorFlow, you have to download the data sets for the test problems. This can be done by simply running the Prepare Data script:

    deepobs_prepare_data.sh

#### Run your Optimizer

The easiest way to use **DeepOBS** with your new optimizer is to write a run script for it. You will only need to write 6 lines of code, specifying which optimizer you want to use and what hyperparameters it has. That is all.

Here is an example using the momentum optimizer in PyTorch:

```python
"""Example run script for PyTorch and the momentum optimizer."""

from torch.optim import SGD
from deepobs import pytorch as pt

optimizer_class = SGD
hyperparams = {"lr": {"type": float},
               "momentum": {"type": float, "default": 0.99},
               "nesterov": {"type": bool, "default": False}}

runner = pt.runners.StandardRunner(optimizer_class, hyperparams)
runner.run(testproblem='quadratic_deep', hyperparams={'lr': 1e-2}, num_epochs=10)
```

Now you are ready to run your optimzier on all test problems of **DeepOBS**, for example to run it on a simple noisy quadratic problem try

    python example_runner.py quadratic_deep --learning_rate 1e-2

#### Further Steps

The next steps in the tutorial and a full recommended step-by-step protocol for benchmarking deep learning optimizers can be found in the [documentation](https://deepobs.readthedocs.io/).

## 🏅 Leaderboard & Baselines

We keep an [online leaderboard](https://deepobs.github.io/#Leaderboard) of our benchmark sets. All entries in the leaderboard are automatically available for comparisons via our [baselines](https://github.com/fsschneider/DeepOBS_Baselines).

ℹ️ If you have an optimizer that you believe should be in the leaderboard let us know!

![Leaderboard](https://raw.githubusercontent.com/fsschneider/DeepOBS/develop/docs/Leaderboard.png "Leaderboard")

## 👨‍👨‍👧‍👦 Contributors

ℹ️ We are always looking for additional support. We hope that **DeepOBS** is the start of a community effort to improve the quality of deep learning optimizer benchmarks. If you find any bugs, stumbling blocks or missing interesting test problems feel free to contact us, create an issue or add a pull request. If you created a new optimizer and tested it with **DeepOBS** please notify us, so that we can include your results in the leaderboard and our baselines.

Many people have contributed to **DeepOBS** and were essential during its development. The list is roughly in chronological order and does not represent the amount of effort put in.

<table>
  <tr>
    <th><a href="https://github.com/lballes"><img alt="Lukas Balles" src="https://avatars0.githubusercontent.com/u/8748569?s=460&amp;v=4" class="contrib" style="border-radius: 50%;"></a>
    <th><a href="https://github.com/philipphennig"><img alt="Philipp Hennigs" src="https://avatars0.githubusercontent.com/u/44397767?s=400&amp;v=4" class="contrib" style="border-radius: 50%;"></a>
    <th><a href="https://github.com/fsschneider"><img alt="Frank Schneider" src="https://avatars0.githubusercontent.com/u/12153723?s=420&amp;v=4" class="contrib" style="border-radius: 50%;"></a>
    <th><a href="https://github.com/abahde"><img alt="Aaron Bahde" src="https://avatars0.githubusercontent.com/u/44397767?s=400&amp;v=4" class="contrib" style="border-radius: 50%;"></a>
    <th><a href="https://github.com/f-dangel"><img alt="Felix Dangel" src="https://avatars0.githubusercontent.com/u/48687646?s=400&amp;v=4" class="contrib" style="border-radius: 50%;"></a>
    <th><a href="https://github.com/prabhuteja12"><img alt="Prabhu Teja Sivaprasad" src="https://avatars0.githubusercontent.com/u/11191577?s=400&amp;v=4" class="contrib" style="border-radius: 50%;"></a>
    <th><a href="https://github.com/florianmai"><img alt="Florian Mai" src="https://avatars0.githubusercontent.com/u/4035329?s=420&amp;v=4" class="contrib" style="border-radius: 50%;"></a>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/lballes">Lukas Balles</a></td>
    <td align="center"><a href="https://github.com/philipphennig">Philipp Hennig</a></td>
    <td align="center"><a href="https://github.com/fsschneider">Frank Schneider</a></td>
    <td align="center"><a href="https://github.com/abahde">Aaron Bahde</a></td>
    <td align="center"><a href="https://github.com/f-dangel">Felix Dangel</a></td>
    <td align="center"><a href="https://github.com/prabhuteja12">Prabhu Teja Sivaprasad</a></td>
    <td align="center"><a href="https://github.com/florianmai">Florian Mai</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://www.is.mpg.de/en">MPI-IS Tübingen</a>, <a href="https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/methods-of-machine-learning/start/">University of Tübingen</a></td>
    <td align="center"><a href="https://www.is.mpg.de/en">MPI-IS Tübingen</a>, <a href="https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/methods-of-machine-learning/start/">University of Tübingen</a></td>
    <td align="center"><a href="https://www.is.mpg.de/en">MPI-IS Tübingen</a>, <a href="https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/methods-of-machine-learning/start/">University of Tübingen</a></td>
    <td align="center"> <a href="https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/methods-of-machine-learning/start/">University of Tübingen</a></td>
    <td align="center"><a href="https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/methods-of-machine-learning/start/">University of Tübingen</a></td>
    <td align="center"><a href="https://www.idiap.ch/en">Idiap Research Institute</a>, <a href="https://www.epfl.ch/en/">EPFL, Switzerland</a></td>
    <td align="center"><a href="https://www.idiap.ch/en">Idiap Research Institute</a>, <a href="https://www.epfl.ch/en/">EPFL, Switzerland</a></td>
  </tr>
  <tr>
    <th>
    <th>
    <th>
    <th>
    <th>
    <th>
    <th>
  </tr>
  <tr>
    <th><a href="https://github.com/SirRob1997"><img alt="Robin Schmidt" src="https://avatars0.githubusercontent.com/u/20804972?s=400&v=4" class="contrib" style="border-radius: 50%;"></a>
    <th>
    <th>
    <th>
    <th>
    <th>
    <th>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/SirRob1997">Robin Schmidt</a></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center"><a href="https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/methods-of-machine-learning/start/">University of Tübingen</a></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
</table>

<!-- Add List with images of contributors (Frank, Lukas,Philipp, Aaron, Felix, Florian Mai, etc.) in chronological order. Also add list of "borrowed sources" (where we got the test problems from):
Here is a list of all authors on relevant research papers that Kaolin borrows code from. Without the efforts of these folks (and their willingness to release their implementations under permissable copyleft licenses), Kaolin would not have been possible.-->

## 📝 Citation

If you use DeepOBS in your work, we would appreciate a reference to our ICLR paper:

[Frank Schneider, Lukas Balles, Philipp Hennig<br/>
**DeepOBS: A Deep Learning Optimizer Benchmark Suite**<br/>
*ICLR 2019*](https://openreview.net/forum?id=rJg6ssC5Y7)

BibTeX entry:

```bibtex
@InProceedings{schneider2018deepobs,
Title = {Deep{OBS}: A Deep Learning Optimizer Benchmark Suite},
Author = {Frank Schneider and Lukas Balles and Philipp Hennig},
Booktitle = {International Conference on Learning Representations},
Year = {2019},
Url = {https://openreview.net/forum?id=rJg6ssC5Y7}
}
```
