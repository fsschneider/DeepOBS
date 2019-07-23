================
Estimate Runtime
================

A convenience script to estimate the run time overhead of a new optimization
method compared to SGD.

By default this script runs SGD as well as the new optimizer ``5`` times for
``3`` epochs on the multi-layer perceptron on MNIST while measuring the time.
It will output the mean run time overhead of the new optimizer for these runs.

Optionally the setup can be changed, by varying the test problem, the number of
epochs, the number of runs, etc. if this allows for a fairer evaluation.

**Usage:**

.. argparse::
   :filename: ../deepobs/scripts/deepobs_estimate_runtime.py
   :func: parse_args
   :prog: deepobs_estimate_runtime.py
