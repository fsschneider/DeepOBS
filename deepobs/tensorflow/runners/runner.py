"""Module implementing StandardRunner."""

from __future__ import print_function

import importlib
import tensorflow as tf
import abc
from .. import config
from .. import testproblems
from . import runner_utils

from deepobs.abstract_runner.abstract_runner import Runner

class TFRunner(Runner, abc.ABC):
    def __init__(self, optimizer_class,hyperparams):

        super(TFRunner, self).__init__(optimizer_class, hyperparams)

    def run(self,
            testproblem,
            batch_size,
            num_epochs,
            random_seed=42,
            data_dir=None,
            output_dir='./results',
            weight_decay=None,
            no_logs=False,
            **training_params
            ):

        # TODO sketch the structure of a runner
        """..."""
        run_folder_name, file_name = self.create_output_directory(testproblem,
                                     num_epochs,
                                     batch_size,
                                     weight_decay,
                                     random_seed,
                                     output_dir,
                                     **training_params)

        if data_dir is not None:
            config.set_data_dir(data_dir)

        tproblem = self.create_testproblem(testproblem, batch_size, weight_decay, random_seed)

        output = self.training(tproblem, num_epochs, **training_params)

        # merge meta data to output dict
        # TODO this step interacts with the Analyzer and should be the same for both methods!
        # TODO Attention! train_params that default are not written to output (e.g. train log interval)!
        output = {'testproblem': testproblem,
                  'batch_size': batch_size,
                  'num_epochs': num_epochs,
                  'random_seed': random_seed,
                  'weight_decay': weight_decay,
                  'optimizer_hyperparams': {**self._optimizer_hyperparams},
                  'training_params': {**training_params},
                  **output}

        if not no_logs:
            self.write_output(output, run_folder_name, file_name)

    @staticmethod
    def create_testproblem(testproblem, batch_size, weight_decay, random_seed):
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
        return tproblem

    # Wrapper functions for the evaluation phase.
    # TODO get rid of test arg and split into two function for clarity?
    @staticmethod
    def evaluate(tproblem, sess, loss, tf_logging, test=True):
        """Computes average loss and accuracy in the evaluation phase."""
        if test:
            sess.run(tproblem.test_init_op)
            msg = "TEST:"
        else:
            sess.run(tproblem.train_eval_init_op)
            msg = "TRAIN:"

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

        # TODO how to abstract the tf logging
#        # Log results to tensorflow summaries
#        if tf_logging:
#            if test:
#                tag = "epoch/test_"
#            else:
#                tag = "epoch/train_"
#            summary = tf.Summary()
#            summary.value.add(tag=tag + "loss_", simple_value=loss_)
#            summary.value.add(tag=tag + "acc_", simple_value=acc_)
#            per_epoch_summary_ = sess.run(per_epoch_summaries)
#            summary_writer.add_summary(per_epoch_summary_,
#                                       len(loss_list) - 1)
#            summary_writer.add_summary(summary, len(loss_list) - 1)
#            summary_writer.flush()
#
        print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss_, acc_))

        # Print and log the results.
        return loss_, acc_


    @abc.abstractmethod
    def training(self, testproblem, num_epochs, **training_params):
        """Must be implemented by subclass. Returns a dict of all captured metrices."""
        return

class TFStandardRunner(TFRunner):
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
        super(TFStandardRunner, self).__init__(optimizer_class, hyperparams)

    # This function is a wrapper around _run() which grabs all non-specified
    # arguments from the command line.
    def training(self,
            tproblem,
            num_epochs,
            lr_sched_epochs=None,
            lr_sched_factors=None,
            train_log_interval=10,
            print_train_iter=None,
            tf_logging=None):

        loss = tf.reduce_mean(tproblem.losses) + tproblem.regularizer

        # Set up the optimizer and create learning rate schedule.
        global_step = tf.Variable(0, trainable=False)

        # this is neccesary to apply the lr_sched later.
        # TODO make this clear
        learning_rate = self._optimizer_hyperparams['learning_rate']
        learning_rate_var = tf.Variable(learning_rate, trainable=False)
        hyperparams = self._optimizer_hyperparams
        hyperparams.pop('learning_rate')

        opt = self._optimizer_class(learning_rate_var, **hyperparams)
        lr_schedule = runner_utils.make_lr_schedule(
            learning_rate, lr_sched_epochs, lr_sched_factors)

        # Call optimizer's minimize on loss to update all variables in the
        # TRAINABLE_VARIABLES collection (with a dependency on performing all ops
        # in the collection UPDATE_OPS collection for batch norm, etc).
        with tf.control_dependencies(
            tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            step = opt.minimize(loss, global_step=global_step)

        # Lists to track train/test loss and accuracy.
        train_losses = []
        test_losses = []
        minibatch_train_losses = []
        train_accuracies = []
        test_accuracies = []

        # Tensorboard summaries
        # TODO how to abstract the tf logging
#        if tf_logging:
#            # per iteration
#            mb_train_loss_summary = tf.summary.scalar(
#                "training/minibatch_train_losses",
#                loss,
#                collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
#            # per epoch
#            lr_summary = tf.summary.scalar(
#                "hyperparams/learning_rate",
#                learning_rate_var,
#                collections=[tf.GraphKeys.SUMMARIES, "per_epoch"])
#            batch_summary = tf.summary.scalar(
#                "hyperparams/batch_size",
#                batch_size,
#                collections=[tf.GraphKeys.SUMMARIES, "per_epoch"])
#
#            per_iter_summaries = tf.summary.merge_all(key="per_iteration")
#            per_epoch_summaries = tf.summary.merge_all(key="per_epoch")
#            summary_writer = tf.summary.FileWriter(directory)

        # Start tensorflow session and initialize variables.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Start of training loop.
        for n in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(n, num_epochs))
            self.evaluate(tproblem, sess, loss, tf_logging, test=False)
            self.evaluate(tproblem, sess, loss, tf_logging, test=True)
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
                        # TODO how to abstract the logging?
#                        if tf_logging:
#                            _, loss_, per_iter_summary_ = sess.run(
#                                [step, loss, per_iter_summaries])
#                            summary_writer.add_summary(per_iter_summary_,
#                                                       sess.run(global_step))
#                        else:
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

        return output
