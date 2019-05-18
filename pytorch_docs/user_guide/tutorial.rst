==============
Simple Example
==============

This super short tutorial shows you an example of how the PyTorch version of DeepOBS can be used to benchmark
the performance of a new optimization method for deep learning.

It aims to show you some basic functions of DeepOBS, by
creating a run script for a new optimizer (we will use Stochastic Gradient Descent with Momentum
as an example here) and running it on a very simple test problem.

Create new Run Script
=====================

The easiest way to use DeepOBS with a new optimizer is to write a run script for
it. This run script will import the optimizer and its hyperparameters. For SGD with Momentum this is

.. literalinclude:: example_run.py 

You can download this :download:`example run script\
<../user_guide/example_run.py>` and use it as a template.

You can create several instances of the runner (with different hyperparameters for the optimizer, e.g. ``lr``) and
run them on the same testproblem. You can then analyze the results and get the best hyperparamater set up.

Analyzing the Runs
==================
The Analyzer class of DeepOBS comes with different functionalities. This is an example script to analyze the above mentioned runs:

.. literalinclude:: example_analyze.py 

You can download this :download:`example Analyzer script\
<../user_guide/example_analyze.py>` and use it as a template.

At first we have to create an Analyzer object with the results folder from the run script. We further have to specify the ``metric`` which should be used to decide whether 
a run was better or worse than others (here we use test accuracies).

We can then print the best settings by calling the method ``print_best_runs()``. We distuingish three different modes of the analysis: ``best``, ``final`` and ``most``. Where ``best`` means the setting of the optimizer that led to the best score of the metric on the whole learning curve. ``Final`` means the setting where the final performance (i.e. after the last epoch) was the best one. ``Most`` means the setting with the most number of runs with different random seeds (you might want to estimate the variance of your optimizer with respect to seeds).

To plot the performance we call the method ``plot_performance()``. It also returns the figure and the axes such that the user could beautify the plots according to her needs.

If the results folder contains the results for several testproblems and/or for several optimizer, the Analyzer will include them automatically in the analysis.
