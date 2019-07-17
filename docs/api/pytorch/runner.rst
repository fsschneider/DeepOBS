============
Runner
============

Runner take care of the actual training process in DeepOBS. They also log
performance statistics such as the loss and accuracy on the test and training
data set.

The output of those runners is saved into ``JSON`` files.

Depending on the design of your optimizer you might have to re-write the training loop. For that, you can create a runner class via inheritance from
deepobs.pytorch.runners.PTRunner. This way, all deepobs internal things of the run (such as setting up the testproblem) are covered by the parent class and you don't have to worry about them. You then just have to write the training() method. You can use the StandardRunner class as a template.

<<<<<<< HEAD:docs/api/runner.rst
If you need to access the model during training, you can do so by using the testproblem instance. Every testproblem instance holds an attribute called 'net' that stores the model.
=======
  runner/standardrunner
  runner/ptrunner
>>>>>>> development:docs/api/pytorch/runner.rst
