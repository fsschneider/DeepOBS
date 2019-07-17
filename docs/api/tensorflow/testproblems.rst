=============
Test Problems
=============

Currently DeepOBS includes twenty-six different test problems. A test problem is
given by a combination of a data set and a model and is characterized by its
loss function.

Each test problem inherits from the same base class with the following signature.

.. currentmodule:: deepobs.tensorflow.testproblems.testproblem

.. autoclass:: TestProblem
    :members:

.. note::
  Some of the test problems described here are based on more general implementations.
  For example the Wide ResNet 40-4 network on Cifar-100 is based on the general
  Wide ResNet architecture which is also implemented. Therefore, it is very easy
  to include new Wide ResNets if necessary.

.. toctree::
  :maxdepth: 2
  :caption: Test Problems

  testproblems/two_d
  testproblems/quadratic
  testproblems/mnist
  testproblems/fmnist
  testproblems/cifar10
  testproblems/cifar100
  testproblems/svhn
  testproblems/imagenet
  testproblems/tolstoi
