=============
Test Problems
=============

Currently the PyTtorch version of DeepOBS includes seven different test problems. A test problem is
given by a combination of a data set and a model and is characterized by its
loss function.

Each test problem inherits from the same base class with the following signature: ``deepobs.pytorch.datasets.dataset.Dataset``