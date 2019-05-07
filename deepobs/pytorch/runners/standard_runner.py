"""Module implementing StandardRunner."""

from __future__ import print_function

import torch
import os
import json
import importlib
from .. import config
from .. import testproblems
from . import runner_utils


class StandardRunner(object):
    """Provides functionality to run optimizers on DeepOBS testproblems including
    the logging of important performance metrics.

    Args:
      optimizer_class: Optimizer class, which should inherit from
          torch.optim.Optimizer.
      hyperparams: A list describing the optimizer's hyperparameters other
          than learning rate. Each entry of the list is a dictionary describing
          one of the hyperparameters. This dictionary is expected to have the
          following two fields:

            - hyperparams["name"] must contain the name of the parameter (i.e.,
              the exact name of the corresponding keyword argument to the
              optimizer class' init function.
            - hyperparams["type"] specifies the type of the parameter (e.g.,
              ``int``, ``float``, ``bool``).

          Optionally, the dictionary can have a third field indexed by the key
          "default", which specifies a default value for the hyperparameter.

    Example
    --------
    >>> optimizer_class = torch.optim.SGD
    >>> hyperparams = [
            {"name": "momentum", "type": float},
            {"name": "nesterov", "type": bool, "default": False}]
    >>> runner = StandardRunner(optimizer_class, hyperparms)

    """

    def __init__(self, optimizer_class, hyperparams):
        """Creates a new StandardRunner.

    Args:
      optimizer_class: Optimizer class, which should inherit from
          torch.optim..Optimizer.
      hyperparams: A list describing the optimizer's hyperparameters other
          than learning rate. Each entry of the list is a dictionary describing
          one of the hyperparameters. This dictionary is expected to have the
          following two fields:
            - hyperparams["name"] must contain the name of the parameter (i.e.,
              the exact name of the corresponding keyword argument to the
              optimizer class' init function.
            - hyperparams["type"] specifies the type of the parameter (e.g.,
              ``int``, ``float``, ``bool``).
          Optionally, the dictionary can have a third field indexed by the key
          "default", which specifies a default value for the hyperparameter.

    Example:
        optimizer_class = torch.optim.SGD
        hyperparams = [
            {"name": "momentum", "type": float},
            {"name": "nesterov", "type": bool, "default": False}]
        runner = StandardRunner(optimizer_class, hyperparms)
    """
        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._hyperparams = hyperparams

    # This function is a wrapper around _run() which grabs all non-specified
    # arguments from the command line.
    def run(self,
            testproblem=None,
            weight_decay=None,
            batch_size=None,
            num_epochs=None,
            learning_rate=None,
            lr_sched_epochs=None,
            lr_sched_factors=None,
            random_seed=None,
            data_dir=None,
            output_dir=None,
            train_log_interval=None,
            print_train_iter=None,
            tf_logging=None,
            no_logs=None,
            **optimizer_hyperparams):
        """Runs a given optimizer on a DeepOBS testproblem.

    This method receives all relevant options to run the optimizer on a DeepOBS
    testproblem, including the hyperparameters of the optimizers, which can be
    passed as keyword arguments (based on the names provided via ``hyperparams``
    in the init function).

    Options which are *not* passed here will
    automatically be added as command line arguments. (Some of those will be
    required, others will have defaults; run the script with the ``--help`` flag
    to see a description of the command line interface.)

    Training statistics (train/test loss/accuracy) are collected and will be
    saved to a ``JSON`` output file, together with metadata.

    Args:
      testproblem (str): Name of a DeepOBS test problem.
      weight_decay (float): The weight decay factor to use.
      batch_size (int): The mini-batch size to use.
      num_epochs (int): The number of epochs to train.
      learning_rate (float): The learning rate to use. This will function as the
          base learning rate when implementing a schedule using
          ``lr_sched_epochs`` and ``lr_sched_factors`` (see below).
      lr_sched_epochs (list): A list of epoch numbers (positive integers) that
          mark learning rate changes. The base learning rate is passed via
          ``learning_rate`` and the factors by which to change are passed via
          ``lr_sched_factors``.
          Example: ``learning_rate=0.3``, ``lr_sched_epochs=[50, 100]``,
          ``lr_sched_factors=[0.1 0.01]`` will start with a learning rate of
          ``0.3``, then decrease to ``0.1*0.3=0.03`` after training for ``50``
          epochs, and decrease to ``0.01*0.3=0.003`` after training for ``100``
          epochs.
      lr_sched_factors (list): A list of factors (floats) by which to change the
          learning rate. The base learning rate has to be passed via
          ``learing_rate`` and the epochs at which to change the learning rate
          have to be passed via ``lr_sched_factors``.
          Example: ``learning_rate=0.3``, ``lr_sched_epochs=[50, 100]``,
          ``lr_sched_factors=[0.1 0.01]`` will start with a learning rate of
          ``0.3``, then decrease to ``0.1*0.3=0.03`` after training for ``50``
          epochs, and decrease to ``0.01*0.3=0.003`` after training for ``100``
          epochs.
      random_seed (int): Random seed to use. If unspecified, it defaults to
          ``42``.
      data_dir (str): Path to the DeepOBS data directory. If unspecified,
          DeepOBS uses its default `/data_deepobs`.
      output_dir (str): Path to the output directory. Within this directory,
          subfolders for the testproblem and the optimizer are automatically
          created. If unspecified, defaults to '/results'.
      train_log_interval (int): Interval of steps at which to log training loss.
          If unspecified it defaults to ``10``.
      print_train_iter (bool): If ``True``, training loss is printed to screen.
          If unspecified it defaults to ``False``.
      no_logs (bool): If ``True`` no ``JSON`` files are created. If unspecified
          it defaults to ``False``.
      optimizer_hyperparams (dict): Keyword arguments for the hyperparameters of
          the optimizer. These are the ones specified in the ``hyperparams``
          dictionary passed to the ``__init__``.
    """

        args = runner_utils.get_arguments(
                self._optimizer_class,
                self._optimizer_name,
                self._hyperparams,
                testproblem,
                weight_decay,
                batch_size,
                num_epochs,
                learning_rate,
                lr_sched_epochs,
                lr_sched_factors,
                random_seed,
                data_dir,
                output_dir,
                train_log_interval,
                print_train_iter,
                no_logs,
                **optimizer_hyperparams)
        self._run(**args)

    def _run(self, testproblem, weight_decay, batch_size, num_epochs,
             learning_rate, lr_sched_epochs, lr_sched_factors, random_seed,
             data_dir, output_dir, train_log_interval, print_train_iter,
             no_logs, **optimizer_hyperparams):
        """Performs the actual run, given all the arguments."""

        # Set data directory of DeepOBS.
        if data_dir is not None:
            config.set_data_dir(data_dir)

        # Find testproblem by name and instantiate with batch size and weight decay.
        try:
            testproblem_mod = importlib.import_module(testproblem)
            testproblem_cls = getattr(testproblem_mod, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)
        if weight_decay is not None:
            tproblem = testproblem_cls(batch_size, weight_decay)
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem.
        torch.manual_seed(random_seed)
        tproblem.set_up()

        # Set up the optimizer and create learning rate schedule.
        opt = self._optimizer_class(tproblem.net.parameters(), lr=learning_rate, **optimizer_hyperparams)
        lr_schedule = runner_utils.make_lr_schedule(optimizer=opt, lr_sched_epochs=lr_sched_epochs, lr_sched_factors=lr_sched_factors)

        # Create output folder
        if not no_logs:
            run_folder_name, file_name = runner_utils.make_run_name(
                weight_decay, batch_size, num_epochs, learning_rate,
                lr_sched_epochs, lr_sched_factors, random_seed,
                **optimizer_hyperparams)
            directory = os.path.join(output_dir, testproblem, self._optimizer_name,
                                     run_folder_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Lists to track train/test loss and accuracy.
        train_losses = []
        test_losses = []
        minibatch_train_losses = []
        train_accuracies = []
        test_accuracies = []


        # Wrapper functions for the evaluation phase.
        def evaluate(test=True):
            """Computes average loss and accuracy in the evaluation phase."""
            if test:
                tproblem.test_init_op()
                msg = "TEST:"
            else:
                tproblem.train_eval_init_op()
                msg = "TRAIN:"

            # evaluation loop over every batch of the corresponding evaluation set
            loss = 0.0
            accuracy = 0.0
            batchCount = 0.0
            while True:
                try:
                    batch_loss, batch_accuracy = tproblem.get_batch_loss_and_accuracy()
                    batchCount += 1.0
                    loss += batch_loss.item()
                    accuracy += batch_accuracy
                except StopIteration:
                    break

            loss /= batchCount
            accuracy /= batchCount

            # if the testproblem has a regularization, add the regularization loss of the current network parameters.
            if hasattr(tproblem, 'get_regularization_loss'):
                loss += tproblem.get_regularization_loss().item()

            if test:
                test_losses.append(loss)
                test_accuracies.append(accuracy)
            else:
                train_losses.append(loss)
                train_accuracies.append(accuracy)
            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss, accuracy))

        # Start of training loop.
        for epoch_count in range(num_epochs + 1):

            # get the next learning rate
            lr_schedule.step()

            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(
                epoch_count, num_epochs))
            evaluate(test=False)
            evaluate(test=True)
            print("********************************")

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            # Training
            if lr_sched_epochs is not None:
                if epoch_count in lr_sched_epochs:
                    print("Setting learning rate to {0}".format(lr_schedule.get_lr()))

            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()
                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()

                    # if the testproblem has a regularization, add the regularization loss.
                    if hasattr(tproblem, 'get_regularization_loss'):
                        regularizer_loss = tproblem.get_regularization_loss()
                        batch_loss += regularizer_loss
                    batch_loss.backward()
                    opt.step()
                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print("Epoch {0:d}, step {1:d}: loss {2:g}".format(epoch_count, batch_count, batch_loss))
                    batch_count += 1
                except StopIteration:
                    break
            # save the model state for sampling
#            if epoch_count % 25 == 0:
#                torch.save(tproblem.net.state_dict(), '/home/isenach/Desktop/Project/models/epoch_' + str(epoch_count))
        # --- End of training loop.

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "optimizer": self._optimizer_name,
            'testproblem': testproblem,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'lr_sched_epochs': lr_sched_epochs,
            'lr_sched_factors': lr_sched_factors,
            'random_seed': random_seed,
            'train_log_interval': train_log_interval,
            'hyperparams': optimizer_hyperparams
        }

        # Dump output into json file.
        if not no_logs:
            with open(os.path.join(directory, file_name + ".json"), "w") as f:
                json.dump(output, f)