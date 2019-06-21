"""Module implementing StandardRunner."""

from __future__ import print_function

import torch
import importlib
import abc
from deepobs import config as global_config
from .. import config
from .. import testproblems
from . import runner_utils
from deepobs.abstract_runner.abstract_runner import Runner
import numpy as np

class PTRunner(Runner):
    def __init__(self, optimizer_class):
        """The abstract class for runner in the pytorch framework.
        Args:
            optimizer_class: The optimizer class of the optimizer that is run on
            the testproblems. Must be a subclass of torch.optim.Optimizer.

            hyperparams (dict): A dict containing the hyperparams for the optimizer_class.

        Methods:
            run: Runs a testproblem with the optimizer_class.
            create_testproblem: Sets up the testproblem.
            evaluate: Evaluates the testproblem model on the testproblem.
            training: An abstract method that has to be overwritten by the subclass.
            It performs the training loop.
        """
        super(PTRunner, self).__init__(optimizer_class)

    @abc.abstractmethod
    def training(self, testproblem, hyperparams, num_epochs, **training_params):
        """Must be implemented by the subclass. Performs the training and stores
        the metrices.
        Args:
            testproblem: instance of the testproblem
            num_epochs (int): number of training epochs
            **training_params (dict): kwargs for the training process

        Must return a dict of the form:

            {'test_losses' : test_losses
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'train_accuracies': train_accuracies,
            'analyzable_training_params': {...}
            }

            where the metrices values are lists that were filled during training
            and the key 'analyzable_training_params' holds a dict of training
            parameters that should be taken into account in the analysis later on.
            These can be, for example, learning rate schedules. Or in the easiest
            case, this dict is empty.
            """
        return

# TODO testproblem and hyperparams have to default to use them in argparse
# how to avoid this? it is more clear if they are required

    def run(self,
            testproblem = None,
            hyperparams = None,
            batch_size = None,
            num_epochs = None,
            random_seed=42,
            data_dir=None,
            output_dir='./results',
            weight_decay=None,
            no_logs=False,
            **training_params
            ):
        """Runs a testproblem with the optimizer_class. Has the following tasks:
            1. setup testproblem
            2. run the training (must be implemented by subclass)
            3. merge and write output

            Input:
                testproblem (str): Name of the testproblem.
                batch_size (int): Mini-batch size for the training data.
                num_epochs (int): The number of training epochs.
                random_seed (int): The torch random seed.
                data_dir (str): The path where the data is stored.
                output_dir (str): Path of the folder where the results are written to.

                weight_decay (float): Regularization factor for the testproblem.
                no_logs (bool): Whether to write the output or not
                **training_params (dict): Kwargs for the training method.
        """

        args = self.parse_args(testproblem,
            hyperparams,
            batch_size,
            num_epochs,
            random_seed,
            data_dir,
            output_dir,
            weight_decay,
            no_logs,
            **training_params)

        # overwrite locals after argparse
        # TODO simplify
        testproblem = args['testproblem']
        hyperparams = args['hyperparams']
        batch_size = args['batch_size']
        num_epochs = args['num_epochs']
        random_seed = args['random_seed']
        data_dir = args['data_dir']
        output_dir = args['output_dir']
        weight_decay = args['weight_decay']
        no_logs = args['weight_decay']
        training_params = args['training_params']

        if batch_size is None:
            batch_size = global_config.get_testproblem_default_setting(testproblem)['batch_size']
        if num_epochs is None:
            num_epochs = global_config.get_testproblem_default_setting(testproblem)['num_epochs']

        if data_dir is not None:
            config.set_data_dir(data_dir)


        tproblem = self.create_testproblem(testproblem, batch_size, weight_decay, random_seed)

        output = self.training(tproblem, hyperparams, num_epochs, **training_params)

        # merge meta data to output dict
        # TODO this step interacts with the Analyzer and should be the same for both frameworks
        output = {'testproblem': testproblem,
                  'batch_size': batch_size,
                  'num_epochs': num_epochs,
                  'random_seed': random_seed,
                  'weight_decay': weight_decay,
                  'optimizer_name': self._optimizer_name,
                  'optimizer_hyperparams': hyperparams,
                  **output}

        if not no_logs:
            run_folder_name, file_name = self.create_output_directory(output_dir, output)
            self.write_output(output, run_folder_name, file_name)

        return output

    @staticmethod
    def create_testproblem(testproblem, batch_size, weight_decay, random_seed):
        """Sets up the testproblem.
        Args:
            testproblem (str): The name of the testproblem.
            batch_size (int): Batch size that is used for training
            weight_decay (float): Regularization factor
            random_seed (int): The random seed of the framework
        Returns:
            tproblem: An instance of deepobs.pytorch.testproblems.testproblem
        """
        # set the seed and GPU determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Find testproblem by name and instantiate with batch size and weight decay.
        try:
            testproblem_mod = importlib.import_module(testproblem)
            testproblem_cls = getattr(testproblem_mod, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)

        # if the user specified a weight decay, use that one
        if weight_decay is not None:
            tproblem = testproblem_cls(batch_size, weight_decay)
        # else use the default of the testproblem
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem.
        tproblem.set_up()
        return tproblem

    # Wrapper functions for the evaluation phase.
    @staticmethod
    def evaluate(tproblem, test=True):
        """Evaluates the performance of the current state of the model
        of the testproblem instance.
        Has to be called in the beggining of every epoch within the
        training method. Returns the losses and accuracies.
        Args:
            tproblem (testproblem): The testproblem instance to evaluate
            test (bool): Whether tproblem is evaluated on the test set.
            If false, it is evaluated in the train evaluation set.
        Returns:
            loss (float): The loss of the current state.
            accuracy (float): The accuracy of the current state.
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

class StandardRunner(PTRunner):
    """A standard runner. Can run a normal training loop with fixed
    hyperparams or a learning rate schedule. It should be used as a template
    to implement custom runners.

    Methods:
        training: Performs the training on a testproblem instance.
    """

    def __init__(self, optimizer_class):

        super(StandardRunner, self).__init__(optimizer_class)

    def training(self,
            tproblem,
            hyperparams,
            num_epochs,
            # the following are the training_params
            lr_sched_epochs=None,
            lr_sched_factors=None,
            train_log_interval=10,
            print_train_iter=False):

        """Input:
                tproblem (testproblem): The testproblem instance to train on.
                num_epochs (int): The number of training epochs.

            **training_params are:
                lr_sched_epochs (list): The epochs where to adjust the learning rate.
                lr_sched_factots (list): The corresponding factors by which to adjust the learning rate.
                train_log_interval (int): When to log the minibatch loss/accuracy.
                print_train_iter (bool): Whether to print the training progress at every train_log_interval

            Returns:
                output (dict): The logged metrices. Is of the form:
                    {'test_losses' : test_losses
                     'train_losses': train_losses,
                     'test_accuracies': test_accuracies,
                     'train_accuracies': train_accuracies,
                     'analyzable_training_params': {...}
                     }

            where the metrices values are lists that were filled during training
            and the key 'analyzable_training_params' holds a dict of training
            parameters that should be taken into account in the analysis later on.
            These can be, for example, learning rate schedules. Or in the easiest
            case, this dict is empty.
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
                lr_schedule.step()
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
            if np.isnan(batch_loss.item()) or np.isinf(batch_loss.item()):
                print('Breaking from run after epoch', str(epoch_count), 'due to wrongly calibrated optimization (Loss is Nan or Inf)')
                break
            else:
                continue

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            # dont need minibatch train losses at the moment
#            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "analyzable_training_params": {
                    "lr_sched_epochs": lr_sched_epochs,
                    "lr_sched_factors": lr_sched_factors
                    }
        }

        return output