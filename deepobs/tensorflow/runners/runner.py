"""Module implementing StandardRunner."""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from . import runner_utils
from copy import deepcopy
from deepobs.abstract_runner import TFRunner


class StandardRunner(TFRunner):

    def __init__(self, optimizer_class, hyperparameter_names):

        super(StandardRunner, self).__init__(optimizer_class, hyperparameter_names)

    def training(self,
                 tproblem, hyperparams, num_epochs, print_train_iter, train_log_interval, tb_log, tb_log_dir):
        """
        asas
        """

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
            if not np.isfinite(loss_):
                train_losses, test_losses, train_accuracies, test_accuracies, minibatch_train_losses = self._abort_routine(n,
                                                                                                   num_epochs,
                                                                                                   train_losses,
                                                                                                   test_losses,
                                                                                                   train_accuracies,
                                                                                                   test_accuracies,
                                                                                                minibatch_train_losses)
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

    def _add_training_params_to_argparse(self, parser, args, training_params):
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
        """
        assa
        """

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
            if not np.isfinite(loss_):
                train_losses, test_losses, train_accuracies, test_accuracies, minibatch_train_losses = self._abort_routine(n,
                                                                                                   num_epochs,
                                                                                                   train_losses,
                                                                                                   test_losses,
                                                                                                   train_accuracies,
                                                                                                   test_accuracies,
                                                                                                minibatch_train_losses)

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
