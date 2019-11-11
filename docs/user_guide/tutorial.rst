==============
Simple Example
==============

This tutorial will show you an example of how DeepOBS can be used to benchmark
the performance of a new optimization method for deep learning.

This simple example aims to show you some basic functions of DeepOBS, by
creating a run script for a new optimizer (we will use the Momentum optimizer
as an example here) and running it on a very simple test problem.

We show this example for **TensorFlow** and **PyTorch** respectively.

Create new Run Script
=====================
The easiest way to use DeepOBS with a new optimizer is to write a run script for
it. This run script will import the optimizer and list its hyperparameters. For the Momentum optimizer in **TensorFlow** this is

.. literalinclude:: ../../examples/runner_momentum_tensorflow.py

You can download this :download:`example run script tensorflow\
<../../examples/runner_momentum_tensorflow.py>` and use it as a template.

For the Momentum optimizer in **PyTorch** it is

.. literalinclude:: ../../examples/runner_momentum_pytorch.py

You can download this :download:`example run script pytorch\
<../../examples/runner_momentum_pytorch.py>` and use it as a template.

The DeepOBS runner needs access to an optimizer class with the same API
as the TensorFlow/PyTorch optimizers and a list of additional hyperparameters for this
new optimizers.

Run new Optimizer
=================

You can now just execute the above mentioned script to run Momentum on the ``quadratic_deep`` test problem.
You can change the arguments in the ``run()`` method to run other test problems, other hyperparameter settings, different number of epochs, etc..
If you want to make the script command line based, you can simply remove all arguments in the ``run()`` method an parse them from the command line.
For **TensorFlow** this would look like this:

.. code-block:: bash

  python runner_momentum_tensorflow.py quadratic_deep --bs 128 --learning_rate 1e-2 --momentum 0.99 --num_epochs 10

We will run it a couple times more this time with different ``learning_rates``

.. code-block:: bash

  python runner_momentum_tensorflow.py quadratic_deep --bs 128 --learning_rate 1e-3 --momentum 0.99 --num_epochs 10
  python runner_momentum_tensorflow.py quadratic_deep --bs 128 --learning_rate 1e-4 --momentum 0.99 --num_epochs 10
  python runner_momentum_tensorflow.py quadratic_deep --bs 128 --learning_rate 1e-5 --momentum 0.99 --num_epochs 10

For **PyTorch** this would look like this:

.. code-block:: bash

  python runner_momentum_pytorch.py quadratic_deep --bs 128 --lr 1e-2 --momentum 0.99 --num_epochs 10

We will run it a couple times more this time with different ``lr``

.. code-block:: bash

  python runner_momentum_pytorch.py quadratic_deep --bs 128 --lr 1e-3 --momentum 0.99 --num_epochs 10
  python runner_momentum_pytorch.py quadratic_deep --bs 128 --lr 1e-4 --momentum 0.99 --num_epochs 10
  python runner_momentum_pytorch.py quadratic_deep --bs 128 --lr 1e-5 --momentum 0.99 --num_epochs 10


Analyzing the Runs
==================

We can use DeepOBS's ``analyzer`` module to automatically find the best hyperparameter setting. First note, that the runner writes the output in a directory tree like:

<results_name>/<testproblem>/<optimizer>/<hyperparameter_setting>/

In the above example, the directory of the run outputs for **TensorFlow** would be:

./results/quadratic_deep/MomentumOptimizer/...

And for **PyTorch**:

./results/quadratic_deep/SGD/...

We pass the path to the optimizer directory to the analyzer functions. This way, we can get the best performance setting, a plot for the corresponding training curve and a plot that visualizes the hyperparameter sensitivity.

For **TensorFlow** and **PyTorch**:

.. literalinclude:: ../../examples/analyzer.py

You need to change the results directory accordingly, i.e. in our example it would be 

.. code-block:: python

   './results/quadratic_deep/SGD'

for TensorFlow (as in the example above) and 

.. code-block:: python

   './results/quadratic_deep/MomentumOptimizer'

for PyTorch.

You can download the script and use it as a template for further analysis:
:download:`example analyze script tensorflow\
<../../examples/analyzer.py>`.

Note that you can also select a reference path (here the deepobs baselines for **TensorFlow**) to plot reference results as well. You can download the latest baselines from `GitHub`_

.. _GitHub: https://github.com/abahde/DeepOBS_Baselines

