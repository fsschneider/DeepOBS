==============================
Tuning Automation
==============================

To address the unfairness that arises from the tuning procedure, we implemented a tuning automation in DeepOBS.
Here, we describe how to use it. We also provide some basic functionalities to monitor the tuning process.
These are not explained here, but can be found in the API section of the :doc:`../api/tuner`. We further
describe a comperative and fair usage of the tuning automation in the :doc:`./suggested_protocol`.


We provide three different Tuner classes: ``GridSearch``, ``RandomSearch`` and ``GP``
(which is a Bayesian optimization method with a Gaussian Process surrogate). You can find detailed information about them
in the API section :doc:`../api/tuner`. We will show all examples in this section for the PyTorch framework.

Grid Search
============

To perform an automated grid search you first have to create the Tuner instance. The optimizer class and its hyperparameters
have to be specified in the same way like for Runners. Additionally, you have to give a dictionary that holds the
discrete values of each hyperparameter. By default, calling ``tune`` will execute the whole tuning process in a sequential
way on the given hardware.

If you want to parallelize the tuning process you can use the method ``generate_commands_script``.
It generates commands than can be send to different nodes. If the format of the command string is not correct for your
training or hyper parameters you have to overwrite the methods ``_generate_kwargs_format_for_command_line`` and
``_generate_hyperparams_format_for_command_line`` of the ``ParallelizedTuner`` accordingly.

.. literalinclude:: ../../example_grid_search.py

You can download this :download:`example\
<../../example_grid_search.py>` and use it as a template.

Random Search
=============
For the random search, you have to give a dictionary that holds the
distributions for each hyperparameter:

.. literalinclude:: ../../example_random_search.py

You can download this :download:`example\
<../../example_random_search.py>` and use it as a template.

Bayesian Optimization (GP)
==========================

The Bayesian optimization method with a Gaussian process surrogate is more complex. At first, you have to specify the
bounds of the suggestions. Additionally, you can set the transformation of the search space. In combination with the
bounds, this can be used for a rescaling of the kernel or for optimization of discrete values:

.. literalinclude:: ../../example_gp_tuner.py

You can download this :download:`example\
<../../example_gp_tuner.py>` and use it as a template. Since Bayesian optimization is sequential by nature, we do not
offer a parallelized version of it.
