"""Module implementing StandardRunner."""

from __future__ import print_function
import numpy as np
import importlib
import tensorflow as tf
import abc
from .. import config
from .. import testproblems
from . import runner_utils
from copy import deepcopy
from deepobs import config as global_config

from deepobs.abstract_runner.abstract_runner import Runner


class TFRunner(Runner):
    def __init__(self, optimizer_class, hyperparameter_names):

        super(TFRunner, self).__init__(optimizer_class, hyperparameter_names)

    def _run(self,
            testproblem = None,
            hyperparams = None,
            batch_size = None,
            num_epochs = None,
            random_seed=None,
            data_dir=None,
            output_dir=None,
            weight_decay=None,
            no_logs=None,
            train_log_interval = None,
            print_train_iter = None,
            tb_log = None,
            tb_log_dir = None,
            **training_params):

        if batch_size is None:
            batch_size = global_config.get_testproblem_default_setting(testproblem)['batch_size']
        if num_epochs is None:
            num_epochs = global_config.get_testproblem_default_setting(testproblem)['num_epochs']

        if data_dir is not None:
            config.set_data_dir(data_dir)

        tproblem = self.create_testproblem(testproblem, batch_size, weight_decay, random_seed)

        output = self.training(tproblem, hyperparams, num_epochs, print_train_iter, train_log_interval, tb_log, tb_log_dir, **training_params)
        output = self._post_process_output(output, 
                                           testproblem, 
                                           batch_size, 
                                           num_epochs, 
                                           random_seed, 
                                           weight_decay, 
                                           hyperparams,
                                           **training_params)
        if not no_logs:
            run_folder_name, file_name = self.create_output_directory(output_dir, output)
            self.write_output(output, run_folder_name, file_name)
        
        return output


    @staticmethod
    def init_summary(loss,
                     learning_rate_var,
                     batch_size,
                     tb_log_dir):
        """Initializes the tensorboard summaries"""
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
        summary_writer = tf.summary.FileWriter(tb_log_dir)
        return per_iter_summaries, per_epoch_summaries, summary_writer

    @staticmethod
    def write_per_epoch_summary(
                                sess,
                                loss_,
                                acc_,
                                current_step,
                                per_epoch_summaries,
                                summary_writer,
                                test=True):
        """Writes the tensorboard epoch summary"""
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

    @staticmethod
    def write_per_iter_summary(
                               sess,
                               per_iter_summaries,
                               summary_writer,
                               current_step):
        """Writes the tensorboard iteration summary"""
        per_iter_summary_ = sess.run(per_iter_summaries)
        summary_writer.add_summary(per_iter_summary_, current_step)


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
        if acc_ != 0.0:
            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss_, acc_))
        else:
            print("{0:s} loss {1:g}".format(msg, loss_))

        # Print and return the results.
        return loss_, acc_


    @abc.abstractmethod
    def training(self, tproblem, hyperparams, num_epochs, print_train_iter, train_log_interval, tb_log, tb_log_dir, **training_params):
        return


class StandardRunner(TFRunner):

    def __init__(self, optimizer_class, hyperparameter_names):

        super(StandardRunner, self).__init__(optimizer_class, hyperparameter_names)

    @staticmethod
    def _add_training_params_to_argparse(parser, args, training_params):
        pass

    def training(self,
                 tproblem, hyperparams, num_epochs, print_train_iter, train_log_interval, tb_log, tb_log_dir):

        loss = tf.reduce_mean(tproblem.losses) + tproblem.regularizer

        # Set up the optimizer and create learning rate schedule.
        global_step = tf.Variable(0, trainable=False)

        # this is neccesary to apply the lr_sched later.
        learning_rate = hyperparams['learning_rate']
        learning_rate_var = tf.Variable(learning_rate, trainable=False)
        hyperparams_ = deepcopy(hyperparams)
        hyperparams_.pop('learning_rate')

        opt = self._optimizer_class(learning_rate_var, **hyperparams_)

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
        if tb_log:
            batch_size = tproblem._batch_size
            per_iter_summaries, per_epoch_summaries, summary_writer = self.init_summary(loss,
                                                                                        learning_rate_var,
                                                                                        batch_size,
                                                                                        tb_log_dir)
        # Start tensorflow session and initialize variables.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Start of training loop.
        for n in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(n, num_epochs))

            loss_, acc_ = self.evaluate(tproblem, sess, loss, test=False)
            if tb_log:
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
            if tb_log:
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
            sess.run(tproblem.train_init_op)
            s = 0
            while True:
                try:
                    if s % train_log_interval == 0:
                        # Training step, with logging if we hit the train_log_interval
                        _, loss_ = sess.run([step, loss])
                        minibatch_train_losses.append(loss_.astype(float))

                        if tb_log:
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

            # break from training if it goes wrong
            if np.isnan(loss_) or np.isinf(loss_):
                train_losses, test_losses, train_accuracies, test_accuracies = self._abort_routine(n,
                                                                                                   num_epochs,
                                                                                                   train_losses,
                                                                                                   test_losses,
                                                                                                   train_accuracies,
                                                                                                   test_accuracies)
                break
            else:
                continue

        sess.close()
        # --- End of training loop.

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accuracies" : train_accuracies,
            "test_accuracies" : test_accuracies,
            "minibatch_train_losses": minibatch_train_losses,
        }
            
        return output


class LearningRateScheduleRunner(TFRunner):

    def __init__(self, optimizer_class, hyperparameter_names):

        super(LearningRateScheduleRunner, self).__init__(optimizer_class, hyperparameter_names)
    @staticmethod
    def _add_training_params_to_argparse(parser, args, training_params):
        try:
            args['lr_sched_epochs'] = training_params['lr_sched_epochs']
        except KeyError:
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

        try:
            args['lr_sched_factors'] = training_params['lr_sched_factors']
        except KeyError:
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

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir,
                 # the following are the training_params
                 lr_sched_epochs=None,
                 lr_sched_factors=None):

        loss = tf.reduce_mean(tproblem.losses) + tproblem.regularizer

        # Set up the optimizer and create learning rate schedule.
        global_step = tf.Variable(0, trainable=False)

        # this is neccesary to apply the lr_sched later.
        learning_rate = hyperparams['learning_rate']
        learning_rate_var = tf.Variable(learning_rate, trainable=False)
        hyperparams_ = deepcopy(hyperparams)
        hyperparams_.pop('learning_rate')

        opt = self._optimizer_class(learning_rate_var, **hyperparams_)
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
        if tb_log:
            batch_size = tproblem._batch_size
            per_iter_summaries, per_epoch_summaries, summary_writer = self.init_summary(loss,
                                                                                        learning_rate_var,
                                                                                        batch_size,
                                                                                        tb_log_dir)
        # Start tensorflow session and initialize variables.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Start of training loop.
        for n in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(n, num_epochs))

            loss_, acc_ = self.evaluate(tproblem, sess, loss, test=False)
            if tb_log:
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
            if tb_log:
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

                        if tb_log:
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

            # break from training if it goes wrong
            if np.isnan(loss_) or np.isinf(loss_):
                train_losses, test_losses, train_accuracies, test_accuracies = self._abort_routine(n,
                                                                                                   num_epochs,
                                                                                                   train_losses,
                                                                                                   test_losses,
                                                                                                   train_accuracies,
                                                                                                   test_accuracies)
                break
            else:
                continue

        sess.close()
        # --- End of training loop.

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "minibatch_train_losses": minibatch_train_losses,
        }

        return output
