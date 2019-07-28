"""Module implementing StandardRunner."""

from __future__ import print_function
from . import runner_utils
from deepobs.abstract_runner import PTRunner
import numpy as np
import warnings


class StandardRunner(PTRunner):
    """A standard runner. Can run a normal training loop with fixed
    hyperparams. It should be used as a template to implement custom runners.

    """

    def __init__(self, optimizer_class, hyperparameter_names):
        super(StandardRunner, self).__init__(optimizer_class, hyperparameter_names)

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir):

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn('Not possible to use tensorboard for pytorch. Reason: ' + e, ImportWarning)
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs+1):
            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(epoch_count, num_epochs))

            loss_, acc_ = self.evaluate(tproblem, test=False)
            train_losses.append(loss_)
            train_accuracies.append(acc_)

            loss_, acc_ = self.evaluate(tproblem, test=True)
            test_losses.append(loss_)
            test_accuracies.append(acc_)

            print("********************************")

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###

            # set to training mode
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
                        if tb_log:
                            summary_writer.add_scalar('loss', batch_loss.item(), global_step)

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                train_losses, test_losses, train_accuracies, test_accuracies, minibatch_train_losses = self._abort_routine(epoch_count,
                                                                                                   num_epochs,
                                                                                                   train_losses,
                                                                                                   test_losses,
                                                                                                   train_accuracies,
                                                                                                   test_accuracies,
                                                                                                minibatch_train_losses)
                break
            else:
                continue

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies
        }

        return output


class LearningRateScheduleRunner(PTRunner):
    """A runner for learning rate schedules. Can run a normal training loop with fixed hyperparams or a learning rate
    schedule. It should be used as a template to implement custom runners.

    Methods:
        training: Performs the training on a testproblem instance.
    """

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
        r"""
        **training_params are:
            lr_sched_epochs (list): The epochs where to adjust the learning rate.
            lr_sched_factors (list): The corresponding factors by which to adjust the learning rate.
            train_log_interval (int): When to log the minibatch loss/accuracy.
            print_train_iter (bool): Whether to print the training progress at every train_log_interval

        Returns:
            output (dict): The logged metrices. Is of the form:
                {'test_losses' : test_losses
                 'train_losses': train_losses,
                 'test_accuracies': test_accuracies,
                 'train_accuracies': train_accuracies
                 }

        where the metrices values are lists that were filled during training.
        """

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)
        if lr_sched_epochs is not None:
            lr_schedule = runner_utils.make_lr_schedule(optimizer=opt, lr_sched_epochs=lr_sched_epochs, lr_sched_factors=lr_sched_factors)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        for epoch_count in range(num_epochs+1):
            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(epoch_count, num_epochs))

            loss_, acc_ = self.evaluate(tproblem, test=False)
            train_losses.append(loss_)
            train_accuracies.append(acc_)

            loss_, acc_ = self.evaluate(tproblem, test=True)
            test_losses.append(loss_)
            test_accuracies.append(acc_)

            print("********************************")

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###
            if lr_sched_epochs is not None:
                # get the next learning rate
                lr_schedule.step(epoch_count)
                if epoch_count in lr_sched_epochs:
                    print("Setting learning rate to {0}".format(lr_schedule.get_lr()))

            # set to training mode
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

            # break from training if it goes wrong
            if not np.isfinite(batch_loss.item()):
                train_losses, test_losses, train_accuracies, test_accuracies = self._abort_routine(epoch_count,
                                                                                                   num_epochs,
                                                                                                   train_losses,
                                                                                                   test_losses,
                                                                                                   train_accuracies,
                                                                                                   test_accuracies)
                break
            else:
                continue

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies
        }

        return output
