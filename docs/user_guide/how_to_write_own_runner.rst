==============================
How to Write Customized Runner
==============================

Some optimizers have special requirements. For example, they need access to the training loop and, therefore, cannot use
DeepOBS as a black box function. Or, the hyperparameters of the optimizer are somewhat special (e.g. other optimizer instances).
For these cases, we give the users the possibility to write their own Runner class.
Here, we describe in more detail what you have to do for that.


Decide for a Framework
======================

Since the latest DeepOBS version comes with TensorFlow and PyTorch implementations you first have to decide on the framework
to use. If you decide for TensorFlow your Runner must inherite from the ``TFRunner`` class. It can be found in the API section
:doc:`../api/tensorflow/runner/TFRunner`.

If you decide for PyTorch your Runner must inherite from the ``PTRunner`` class. It can be found in the API section
:doc:`../api/pytorch/runner/PTRunner`

Implement the Training Loop
============================

The most import implementation for your customized runner is the method ``training`` which runs the training loop
on the testproblem. Its basic signature can be found in the API section for :doc:`../api/tensorflow/runner/TFRunner`
and :doc:`../api/pytorch/runner/PTRunner` respectively. Concrete example implementations
can be found in the Runner classes that come with DeepOBS. We recommend copying one of those and adapt it to your needs.
In principle, simply make sure that the output dictionary is filled with the metrices ``test_accuracies``,
``valid_accuracies``, ``train_accuracies``, ``test_losses``, ``valid_losses`` and ``tain_losses`` during training.
Additionally, we distinguish between ``hyperparameters`` (which are the parameters that are used to initialize
the optimizer) and ``training parameters`` (which are used as additional keyword arguments in the training loop).

For the PyTorch version we would like to give some useful hints:

1. A ``deepobs.pytorch.testproblems.testproblem`` instance holds the attribute ``net`` which is the model that is to be trained.
This way, you have full access to the model parameters during training.

2. Somewhat counterintuitively, we implemented a method ``get_batch_loss_and_accuracy`` for each testproblem. This method
gets the next batch of the training set and evaluates the forward path. We implemented a closure such that you can
call the forward path several times within the trainig loop (e.g. a second time after a parameter update). For this,
simply set the argument ``return_forward_func = True`` of ``get_batch_loss_and_accuracy``.

3. A ``deepobs.pytorch.testproblems.testproblem`` instance holds the attribute ``regularization_groups``. It can be used
to modify the way your optimizer deals with the regularization.

Read in Hyperparameters and Training Parameters from the Command Line
=====================================================================
To use your Runner scripts from the command line, you have to specify the way the hyper and training parameters
should be read in by argparse. For that, you can overwrite the methods ``_add_training_params_to_argparse`` and
``_add_hyperparams_to_argparse``. For both frameworks, examples can be found in the ``LearningRateScheduleRunner``.

Specify How the Hyperparameters and Training Parameters Should Be Added to the Run Name
=======================================================================================
Each individual run ends with writing the output to a well structured directory tree. This is important for later analysis
of the results. To specify how your hyper and training parameters should be used for the naming of the setting
directories, you have to overwrite the methods ``_add_training_params_to_output_dir_name`` and
``_add_hyperparams_to_output_dir_name``. For both frameworks, examples can be found in the ``LearningRateScheduleRunner``.
