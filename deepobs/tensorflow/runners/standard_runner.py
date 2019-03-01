"""Module implementing StandardRunner."""

from __future__ import print_function

import argparse
import os
import json
import importlib
import tensorflow as tf

from .. import config
from .. import testproblems
from . import runner_utils


class StandardRunner(object):
    """Provides functionality to run optimizers on DeepOBS testproblems including
    the logging of important performance metrics.

    Args:
      optimizer_class: Optimizer class, which should inherit from
          tf.train.Optimizer and/or obey the same interface for ``.minimize()``.
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
    >>> optimizer_class = tf.train.MomentumOptimizer
    >>> hyperparams = [
            {"name": "momentum", "type": float},
            {"name": "use_nesterov", "type": bool, "default": False}]
    >>> runner = StandardRunner(optimizer_class, hyperparms)

    """

    def __init__(self, optimizer_class, hyperparams):
        """Creates a new StandardRunner.

    Args:
      optimizer_class: Optimizer class, which should inherit from
          tf.train.Optimizer and/or obey the same interface for ``.minimize()``.
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
        optimizer_class = tf.train.MomentumOptimizer
        hyperparams = [
            {"name": "momentum", "type": float},
            {"name": "use_nesterov", "type": bool, "default": False}]
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
    saved to a ``JSON`` output file, together with metadata. The training
    statistics can optionally also be saved in TensorFlow output files and read
    during training using `Tensorboard`.

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
      tf_logging (bool): If ``True`` log all statistics with tensorflow summaries,
          which can be viewed in real time with tensorboard. If unspecified it
          defaults to ``False``.
      no_logs (bool): If ``True`` no ``JSON`` files are created. If unspecified
          it defaults to ``False``.
      optimizer_hyperparams (dict): Keyword arguments for the hyperparameters of
          the optimizer. These are the ones specified in the ``hyperparams``
          dictionary passed to the ``__init__``.
    """
        # We will go through all the arguments, check whether they have been passed
        # to this function. If yes, we collect the (name, value) pairs  in ``args``.
        # If not, we add corresponding command line arguments.
        args = {}
        parser = argparse.ArgumentParser(
            description="Run {0:s} on a DeepOBS test problem.".format(
                self._optimizer_name))

        if testproblem is None:
            parser.add_argument(
                "testproblem",
                help="""Name of the DeepOBS testproblem
          (e.g. 'cifar10_3c3d'""")
        else:
            args["testproblem"] = testproblem

        if weight_decay is None:
            parser.add_argument(
                "--weight_decay",
                "--wd",
                type=float,
                help="""Factor
          used for the weight_deacy. If not given, the default weight decay for
          this model is used. Note that not all models use weight decay and this
          value will be ignored in such a case.""")
        else:
            args["weight_decay"] = weight_decay

        if batch_size is None:
            parser.add_argument(
                "--batch_size",
                "--bs",
                required=True,
                type=int,
                help="The batch size (positive integer).")
        else:
            args["batch_size"] = batch_size

        if num_epochs is None:
            parser.add_argument(
                "-N",
                "--num_epochs",
                required=True,
                type=int,
                help="Total number of training epochs.")
        else:
            args["num_epochs"] = num_epochs

        if learning_rate is None:
            parser.add_argument(
                "--learning_rate",
                "--lr",
                required=True,
                type=float,
                help=
                """Learning rate (positive float) to use. Can be used as the base
          of a learning rate schedule when used in conjunction with
          --lr_sched_epochs and --lr_sched_factors.""")
        else:
            args["learning_rate"] = learning_rate

        if lr_sched_epochs is None:
            parser.add_argument(
                "--lr_sched_epochs",
                nargs="+",
                type=int,
                help="""One or more epoch numbers (positive integers) that mark
          learning rate changes. The base learning rate has to be passed via
          '--learing_rate' and the factors by which to change have to be passed
          via '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
          --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
          then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
          decrease to 0.01*0.3=0.003' after training for 100 epochs.""")
        else:
            args["lr_sched_epochs"] = lr_sched_epochs

        if lr_sched_factors is None:
            parser.add_argument(
                "--lr_sched_factors",
                nargs="+",
                type=float,
                help=
                """One or more factors (floats) by which to change the learning
          rate. The base learning rate has to be passed via '--learing_rate' and
          the epochs at which to change the learning rate have to be passed via
          '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
          --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
          then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
          decrease to 0.01*0.3=0.003' after training for 100 epochs.""")
        else:
            args["lr_sched_factors"] = lr_sched_factors

        if random_seed is None:
            parser.add_argument(
                "-r",
                "--random_seed",
                type=int,
                default=42,
                help="An integer to set as tensorflow's random seed.")
        else:
            args["random_seed"] = random_seed

        if data_dir is None:
            parser.add_argument(
                "--data_dir",
                help="""Path to the base data dir. If
      not specified, DeepOBS uses its default.""")
        else:
            args["data_dir"] = data_dir

        if output_dir is None:
            parser.add_argument(
                "--output_dir",
                type=str,
                default="results",
                help="""Path to the base directory in which output files will be
          stored. Results will automatically be sorted into subdirectories of
          the form 'testproblem/optimizer'.""")
        else:
            args["output_dir"] = output_dir

        if train_log_interval is None:
            parser.add_argument(
                "--train_log_interval",
                type=int,
                default=10,
                help="Interval of steps at which training loss is logged.")
        else:
            args["train_log_interval"] = train_log_interval

        if print_train_iter is None:
            parser.add_argument(
                "--print_train_iter",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to print mini-batch training loss to
          stdout on each (logged) interation.""")
        else:
            args["print_train_iter"] = print_train_iter

        if tf_logging is None:
            parser.add_argument(
                "--tf_logging",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to log statistics using tensorflow
          (to view in tensorboard).""")
        else:
            args["tf_logging"] = tf_logging

        if no_logs is None:
            parser.add_argument(
                "--no_logs",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to not save any json logging files.""")
        else:
            args["no_logs"] = no_logs

        # Optimizer hyperparams
        for hp in self._hyperparams:
            hp_name = hp["name"]
            if hp_name in optimizer_hyperparams:
                args[hp_name] = optimizer_hyperparams[hp_name]
            else:  # hp_name not in optimizer_hyperparams
                hp_type = hp["type"]
                if "default" in hp:
                    hp_default = hp["default"]
                    parser.add_argument(
                        "--{0:s}".format(hp_name),
                        type=hp_type,
                        default=hp_default,
                        help="""Hyperparameter {0:s} of {1:s} ({2:s};
              defaults to {3:s}).""".format(hp_name, self._optimizer_name,
                                            str(hp_type), str(hp_default)))
                else:
                    parser.add_argument(
                        "--{0:s}".format(hp_name),
                        type=hp_type,
                        required=True,
                        help="Hyperparameter {0:s} of {1:s} ({2:s}).".format(
                            hp_name, self._optimizer_name, str(hp_type)))

        # Get the command line arguments and add them to the ``args`` dict. Then
        # call the _run function with those arguments.
        cmdline_args = vars(parser.parse_args())
        args.update(cmdline_args)
        self._run(**args)

    def _run(self, testproblem, weight_decay, batch_size, num_epochs,
             learning_rate, lr_sched_epochs, lr_sched_factors, random_seed,
             data_dir, output_dir, train_log_interval, print_train_iter,
             tf_logging, no_logs, **optimizer_hyperparams):
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
        tf.reset_default_graph()
        tf.set_random_seed(random_seed)
        tproblem.set_up()
        loss = tf.reduce_mean(tproblem.losses) + tproblem.regularizer

        # Set up the optimizer and create learning rate schedule.
        global_step = tf.Variable(0, trainable=False)
        learning_rate_var = tf.Variable(learning_rate, trainable=False)
        opt = self._optimizer_class(learning_rate_var, **optimizer_hyperparams)
        lr_schedule = runner_utils.make_lr_schedule(
            learning_rate, lr_sched_epochs, lr_sched_factors)

        # Call optimizer's minimize on loss to update all variables in the
        # TRAINABLE_VARIABLES collection (with a dependency on performing all ops
        # in the collection UPDATE_OPS collection for batch norm, etc).
        with tf.control_dependencies(
            tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            step = opt.minimize(loss, global_step=global_step)

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

        # Tensorboard summaries
        if tf_logging:
            # per iteration
            mb_train_loss_summary = tf.summary.scalar(
                "training/minibatch_train_losses",
                loss,
                collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
            # per epoch
            lr_summary = tf.summary.scalar(
                "hyperparams/learning_rate",
                learning_rate_var,
                collections=[tf.GraphKeys.SUMMARIES, "per_epoch"])
            batch_summary = tf.summary.scalar(
                "hyperparams/batch_size",
                batch_size,
                collections=[tf.GraphKeys.SUMMARIES, "per_epoch"])

            per_iter_summaries = tf.summary.merge_all(key="per_iteration")
            per_epoch_summaries = tf.summary.merge_all(key="per_epoch")
            summary_writer = tf.summary.FileWriter(directory)

        # Start tensorflow session and initialize variables.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Wrapper functions for the evaluation phase.
        def evaluate(test=True):
            """Computes average loss and accuracy in the evaluation phase."""
            if test:
                sess.run(tproblem.test_init_op)
                msg = "TEST:"
                loss_list = test_losses
                acc_list = test_accuracies
            else:
                sess.run(tproblem.train_eval_init_op)
                msg = "TRAIN:"
                loss_list = train_losses
                acc_list = train_accuracies

            # Compute average loss and (if applicable) accuracy.
            loss_ = 0.0
            num_iters = 0.0
            acc_ = 0.0
            if tproblem.accuracy is not None:
                while True:
                    try:
                        l_, a_ = sess.run([loss, tproblem.accuracy])
                        loss_ += l_
                        acc_ += a_
                        num_iters += 1.0
                    except tf.errors.OutOfRangeError:
                        break
            else:  # accuracy is None
                acc_ = 0.0
                while True:
                    try:
                        l_ = sess.run(loss)
                        loss_ += l_
                        num_iters += 1.0
                    except tf.errors.OutOfRangeError:
                        break
            loss_ /= num_iters
            acc_ /= num_iters

            # Print and log the results.
            loss_list.append(loss_)
            acc_list.append(acc_)
            # Log results to tensorflow summaries
            if tf_logging:
                if test:
                    tag = "epoch/test_"
                else:
                    tag = "epoch/train_"
                summary = tf.Summary()
                summary.value.add(tag=tag + "loss_", simple_value=loss_)
                summary.value.add(tag=tag + "acc_", simple_value=acc_)
                per_epoch_summary_ = sess.run(per_epoch_summaries)
                summary_writer.add_summary(per_epoch_summary_,
                                           len(loss_list) - 1)
                summary_writer.add_summary(summary, len(loss_list) - 1)
                summary_writer.flush()

            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss_, acc_))

        # Start of training loop.
        for n in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(
                n, num_epochs))
            evaluate(test=False)
            evaluate(test=True)
            print("********************************")

            # Break from train loop after the last round of evaluation
            if n == num_epochs:
                break

            # Training
            if n in lr_schedule:
                sess.run(learning_rate_var.assign(lr_schedule[n]))
                print("Setting learning rate to {0:f}".format(lr_schedule[n]))
            sess.run(tproblem.train_init_op)
            s = 0
            while True:
                try:
                    # Training step, with logging if we hit the train_log_interval
                    if s % train_log_interval == 0:
                        if tf_logging:
                            _, loss_, per_iter_summary_ = sess.run(
                                [step, loss, per_iter_summaries])
                            summary_writer.add_summary(per_iter_summary_,
                                                       sess.run(global_step))
                        else:
                            _, loss_ = sess.run([step, loss])
                        minibatch_train_losses.append(loss_.astype(float))
                        if print_train_iter:
                            print("Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                n, s, loss_))
                    else:
                        sess.run(step)
                    s += 1
                except tf.errors.OutOfRangeError:
                    break

        sess.close()
        # --- End of training loop.

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses
        }
        if tproblem.accuracy is not None:
            output["train_accuracies"] = train_accuracies
            output["test_accuracies"] = test_accuracies

        # Put all run parameters into output dictionary.
        output["optimizer"] = self._optimizer_name
        output["testproblem"] = testproblem
        output["weight_decay"] = weight_decay
        output["batch_size"] = batch_size
        output["num_epochs"] = num_epochs
        output["learning_rate"] = learning_rate
        output["lr_sched_epochs"] = lr_sched_epochs
        output["lr_sched_factors"] = lr_sched_factors
        output["random_seed"] = random_seed
        output["train_log_interval"] = train_log_interval

        # Add optimizer hyperparameters as a sub-dictionary.
        output["hyperparams"] = optimizer_hyperparams

        # Dump output into json file.
        if not no_logs:
            with open(os.path.join(directory, file_name + ".json"), "w") as f:
                json.dump(output, f)
