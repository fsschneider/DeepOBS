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

        self._run_folder_name, file_name = self.create_output_directory(testproblem,
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
            self.write_output(output, self._run_folder_name, file_name)

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
    def evaluate(tproblem, sess, loss, test=True):
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
#
        print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss_, acc_))

        # Print and return the results.
        return loss_, acc_


    @abc.abstractmethod
    def training(self, testproblem, num_epochs, **training_params):
        """Must be implemented by subclass. Returns a dict of all captured metrices."""
        return

class TFStandardRunner(TFRunner):


    def __init__(self, optimizer_class, hyperparams):

        super(TFStandardRunner, self).__init__(optimizer_class, hyperparams)

    def init_summary(self, loss,
                     learning_rate_var,
                     batch_size,
                     log_dir):
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
        summary_writer = tf.summary.FileWriter(log_dir)
        return per_iter_summaries, per_epoch_summaries, summary_writer

    def write_per_epoch_summary(self,
                                sess,
                                loss_,
                                acc_,
                                current_step,
                                per_epoch_summaries,
                                summary_writer,
                                test=True):
        if test:
            tag = "epoch/test_"
        else:
            tag = "epoch/train_"
        summary = tf.Summary()
        summary.value.add(tag=tag + "loss_", simple_value=loss_)
        summary.value.add(tag=tag + "acc_", simple_value=acc_)
        per_epoch_summary_ = sess.run(per_epoch_summaries)
        summary_writer.add_summary(per_epoch_summary_,
                                   current_step)
        summary_writer.add_summary(summary, current_step)
        summary_writer.flush()
        return

    def write_per_iter_summary(self,
                               sess,
                               per_iter_summaries,
                               summary_writer,
                               current_step):
        per_iter_summary_ = sess.run(per_iter_summaries)
        summary_writer.add_summary(per_iter_summary_, current_step)

    # This function is a wrapper around _run() which grabs all non-specified
    # arguments from the command line.
    def training(self,
            tproblem,
            num_epochs,
            lr_sched_epochs=None,
            lr_sched_factors=None,
            train_log_interval=10,
            print_train_iter=False,
            tf_logging=False,
            tf_logging_dir='./tensorboard_logs'):

        # TODO abstract loss
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
        if tf_logging:
            batch_size = tproblem._batch_size
            per_iter_summaries, per_epoch_summaries, summary_writer = self.init_summary(loss,
                                                                                        learning_rate_var,
                                                                                        batch_size,
                                                                                        tf_logging_dir)
        # Start tensorflow session and initialize variables.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Start of training loop.
        for n in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(n, num_epochs))

            loss_, acc_ = self.evaluate(tproblem, sess, loss, test=False)
            if tf_logging:
                current_step = len(train_losses)
                self.write_per_epoch_summary(sess,
                                         loss_,
                                         acc_,
                                         current_step,
                                         per_epoch_summaries,
                                         summary_writer,
                                         test=False)
            train_losses.append(loss_)
            train_accuracies.append(acc_)

            loss_, acc_ = self.evaluate(tproblem, sess, loss, test=True)
            if tf_logging:
                current_step = len(test_losses)
                self.write_per_epoch_summary(sess,
                                         loss_,
                                         acc_,
                                         current_step,
                                         per_epoch_summaries,
                                         summary_writer,
                                         test=True)
            test_losses.append(loss_)
            test_accuracies.append(acc_)
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
                    if s % train_log_interval == 0:
                        # Training step, with logging if we hit the train_log_interval
                        _, loss_ = sess.run([step, loss])
                        minibatch_train_losses.append(loss_.astype(float))

                        if tf_logging:
                            current_step = sess.run(global_step)
                            self.write_per_iter_summary(sess,
                                                        per_iter_summaries,
                                                        summary_writer,
                                                        current_step)

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
