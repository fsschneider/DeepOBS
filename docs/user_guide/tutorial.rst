==============
Simple Example
==============

This super short tutorial shows you an example of how the PyTorch version of DeepOBS can be used to benchmark
the performance of a new optimization method for deep learning.

It aims to show you some basic functions of DeepOBS, by
creating a run script for a new optimizer (we will use Stochastic Gradient Descent with Momentum
as an example here) and running it on a very simple test problem.

We show this example for TensorFlow. If you use PyTorch, the only difference is the optimizer class (torch.optim.SGD) and the hyperparameter names. 

Create new Run Script
=====================
The easiest way to use DeepOBS with a new optimizer is to write a run script for it. This run script will import the optimizer and list its hyperparameters. For the Momentum optimizer this is simply

.. literalinclude:: ../../example_momentum_runner.py

You can download this :download:`example run script\
<../../example_momentum_runner.py>` and use it as a template.

The DeepOBS runner (Line 7) needs access to an optimizer class with the same API
as the TensorFlow/PyTorch optimizers and a list of additional hyperparameters for this
new optimizers.

This run script is now fully command line based and is able to access all the
test problems (and other options) of DeepOBS while also allowing to specify the
new optimizers hyperparameters.


Run new Optimizer
=================

Assuming the run script (from the previous section) is called
``example_momentum_runner.py`` we can use it to run the Momentum optimizer on one of the
test problems on DeepOBS:

.. code-block:: bash

  python example_momentum_runner.py quadratic_deep --bs 128 --lr 1e-2 --momentum 0.99 --num_epochs 10

We will run it a couple times more this time with different ``learning_rates``

.. code-block:: bash

  python example_momentum_runner.py quadratic_deep --bs 128 --learning_rate 1e-3 --momentum 0.99 --num_epochs 10
  python example_momentum_runner.py quadratic_deep --bs 128 --learning_rate 1e-4 --momentum 0.99 --num_epochs 10
  python example_momentum_runner.py quadratic_deep --bs 128 --learning_rate 1e-5 --momentum 0.99 --num_epochs 10


Get best Run
============

We can use DeepOBS's ``analyzer`` module to automatically find the best hyperparameter setting. First note, that the runner writes the output in a directory tree like:

<results_name>/<testproblem>/<optimizer>/<hyperparameter_setting>/

In the above example, the directory of the run outputs will be:

./results/quadratic_deep/MomentumOptimizer/...

We pass the path to the optimizer directory to the analyzer functions. This way, we can get the best performance setting, a plot for the corresponding training curve and a plot that visualizes the hyperparameter sensitivity:

.. literalinclude:: ../../example_analyze.py

Note that you can also select a reference path (here the deepobs baselines) to plot reference results as well.
