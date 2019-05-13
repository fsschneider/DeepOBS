"""Module implementing StandardRunner."""

from __future__ import print_function

import torch
import importlib
import abc
from .. import config
from .. import testproblems
from . import runner_utils
from deepobs.abstract_runner.abstract_runner import Runner

class PTRunner(Runner, abc.ABC):
    def __init__(self, optimizer_class,hyperparams):

        super(PTRunner, self).__init__(optimizer_class, hyperparams)

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
        torch.manual_seed(random_seed)
        tproblem.set_up()
        return tproblem

    # Wrapper functions for the evaluation phase.
    # TODO get rid of test arg and split into two function for clarity?
    @staticmethod
    def evaluate(tproblem, test=True):
        """Evaluates the performance of the current state of the network of the testproblem instance.
        Has to be called in the beggining of every epoch. Returns the losses and accuracies of the current
        state for the test or train_eval_state
        """

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

        print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss, accuracy))

        return loss, accuracy

    @abc.abstractmethod
    def training(self, testproblem, num_epochs, **training_params):
        """Must be implemented by subclass. Returns a dict of all captured metrices."""
        return

class StandardRunner(PTRunner):

    def __init__(self, optimizer_class, hyperparams):

        super(StandardRunner, self).__init__(optimizer_class, hyperparams)

    def training(self,
            tproblem,
            num_epochs,
            lr_sched_epochs=None,
            lr_sched_factors=None,
            train_log_interval=10,
            print_train_iter=False):

        # TODO make clear that this runner is indeed normal if no schedule is applied
        opt = self._optimizer_class(tproblem.net.parameters(), **self._optimizer_hyperparams)
        lr_schedule = runner_utils.make_lr_schedule(optimizer=opt, lr_sched_epochs=lr_sched_epochs, lr_sched_factors=lr_sched_factors)

        # Lists to track train/test loss and accuracy.
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

            # get the next learning rate
            lr_schedule.step()

            # Training
            if lr_sched_epochs is not None:
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

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
        }

        return output