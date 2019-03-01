============
Runner
============

Runner take care of the actual training process in DeepOBS. They also log
performance statistics such as the loss and accuracy on the test and training
data set.

The output of those runners is saved into ``JSON`` files and optionally also
TensorFlow output files that can be plotted in real-time using `Tensorboard`.

.. toctree::
  :maxdepth: 2
  :caption: Runner

  runner/standardrunner
