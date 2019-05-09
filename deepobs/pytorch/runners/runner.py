"""Module implementing StandardRunner."""

from __future__ import print_function

import torch
import importlib
from .. import config
from .. import testproblems
from . import runner_utils
from deepobs.abstract_runner.abstract_runner import Runner

class PTRunner(Runner):
    def __init__(self, optimizer_class,hyperparams):

        super(PTRunner, self).__init__(optimizer_class, hyperparams)

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
            no_logs=None):

        # read the pt specific arguments from the command line
        args = runner_utils.read_args(
            self._optimizer_class,
            self._optimizer_name,
            testproblem,
            weight_decay,
            batch_size,
            num_epochs,
            learning_rate,
            data_dir,
            lr_sched_epochs,
            lr_sched_factors,
            random_seed,
            output_dir,
            print_train_iter,
            no_logs,
            train_log_interval)

        self._run(**args)

    def _run(self):
        """A runner specific method that calculates the metrices test/train losses/accuracies"""
        raise NotImplementedError(
            """'PTRunner' is an abstract base class, please use
        one of the sub-classes.""")

class StandardRunner(PTRunner):


    def __init__(self, optimizer_class, hyperparams):

        super(StandardRunner, self).__init__(optimizer_class, hyperparams)

    def _run(self,
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
            no_logs=None):

        """Performs the training process."""

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
        opt = self._optimizer_class(tproblem.net.parameters(), lr=learning_rate, **self._hyperparams)
        lr_schedule = runner_utils.make_lr_schedule(optimizer=opt, lr_sched_epochs=lr_sched_epochs, lr_sched_factors=lr_sched_factors)



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
                if epoch_count in self.lr_sched_epochs:
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
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hyperparams': self._hyperparams,
            'testproblem': testproblem,
            'weight_decay': weight_decay,
            'lr_sched_epochs': lr_sched_epochs,
            'lr_sched_factors': lr_sched_factors,
            'random_seed': random_seed,
            'train_log_interval': train_log_interval,
        }

        if not no_logs:
            directory, file_name = self.create_output_folder(
                             self._hyperparams,
                             testproblem,
                             output_dir,
                             weight_decay,
                             batch_size,
                             num_epochs,
                             learning_rate,
                             lr_sched_epochs,
                             lr_sched_factors,
                             random_seed)

            self.write_output(output, directory, file_name)