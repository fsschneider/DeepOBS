==================
Suggested Protocol
==================

Here we provide a suggested protocol for more rigorously benchmarking deep
learning optimizer. It follows the same steps as the baseline results presented
in the `DeepOBS`_ paper

.. _DeepOBS: https://openreview.net/forum?id=rJg6ssC5Y7

Create new Run Script
=====================

In order to benchmark a new optimization method a new run script has to be
written. A more detailed description can be found in the :doc:`tutorial` and
the the API section for TensorFlow (:doc:`../api/tensorflow/runner/standardrunner`) and PyTorch (:doc:`../api/tensorflow/runner/standardrunner`) respecitvely
Essentially, all which is needed is the optimizer itself and a list of its hyperparameters. For example
for the Momentum optimizer in **Tensorlow** this will be:
.. literalinclude:: ../../example_momentum_runner_tensorflow.py
And in **PyTorch**:
.. literalinclude:: ../../example_momentum_runner_pytorch.py

Hyperparameter Search
=====================

Once the optimizer has been defined it is recommended to do a hyperparameter
search for each test problem. For optimizers with only the ``learning rate`` as
a free parameter a simple grid search can be done.

For the baselines, we tuned the ``learning rate`` for each optimizer and test
problem individually, by evaluating on a logarithmic grid from ``10e−5``
to ``10e2`` with ``36`` samples. If the same tuning method is used for a new
optimizer no re-running of the baselines is needed saving valuable
computational budget.

.. NOTE::
  Alternatively, this DeepOBS version comes with a tuning automation module: :doc:`../api/tuner`.
  However, at the moment we do not provide a suggested protocol of how to use it.


Repeated Runs with best Setting
===============================

In order to get a sense of the optimziers consistency, we suggest repeating
runs with the best hyperparameter setting multiple times. This allows an
assessment of the variance of the optimizer's performance.

For the baselines we determined the best learning rate looking at the final
performance of each run, which can be done using :doc:`../api/analyzer`,
and then running the best performing setting again using ten different random
seeds.

Plot Results
============

To visualize the final results, the user can use the :doc:`../api/analyzer` API.
For most functionalities, if the path to the baseline folder is given, DeepOBS will automatically compare
the results with the baselines for ``SGD``, ``Momentum``, and ``Adam``.


