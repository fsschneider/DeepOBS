"""Module implementing StandardRunner."""

from __future__ import print_function

import os
import abc
import importlib
import warnings
from copy import deepcopy
from random import seed

import numpy as np
import torch

import matplotlib.pyplot as plt
import torchvision.utils as vutils

from deepobs import config as global_config
from deepobs.abstract_runner.abstract_runner import Runner

from .. import config, testproblems
from . import runner_utils


class PTRunner(Runner):
    """The abstract class for runner in the pytorch framework."""

    def __init__(self, optimizer_class, hyperparameter_names):
        super(PTRunner, self).__init__(optimizer_class, hyperparameter_names)

    @abc.abstractmethod
    def training(
            self,
            tproblem,
            hyperparams,
            num_epochs,
            print_train_iter,
            train_log_interval,
            tb_log,
            tb_log_dir,
            **training_params
    ):
        return

    @abc.abstractmethod
    def training_gan(
            self,
            tproblem,
            hyperparams,
            num_epochs,
            print_train_iter,
            train_log_interval,
            tb_log,
            tb_log_dir,
            eval_interval,
            **training_params
    ):
        return

    @staticmethod
    def create_testproblem(testproblem, batch_size, l2_reg, random_seed):
        """Sets up the deepobs.pytorch.testproblems.testproblem instance.

        Args:
            testproblem (str): The name of the testproblem.
            batch_size (int): Batch size that is used for training
            l2_reg (float): Regularization factor
            random_seed (int): The random seed of the framework

        Returns:
            deepobs.pytorch.testproblems.testproblem: An instance of deepobs.pytorch.testproblems.testproblem
        """
        # set the seed and GPU determinism
        if config.get_is_deterministic():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Find testproblem by name and instantiate with batch size and L2-regularization.
        try:
            testproblem_mod = importlib.import_module(testproblem)
            testproblem_cls = getattr(testproblem_mod, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)

        # if the user specified L2-regularization, use that one
        if l2_reg is not None:
            tproblem = testproblem_cls(batch_size, l2_reg)
        # else use the default of the testproblem
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem.
        tproblem.set_up()
        return tproblem

    # Wrapper functions for the evaluation phase.
    @staticmethod
    def evaluate(tproblem, phase):
        """Evaluates the performance of the current state of the model
        of the testproblem instance.
        Has to be called in the beggining of every epoch within the
        training method. Returns the losses and accuracies.

        Args:
            tproblem (testproblem): The testproblem instance to evaluate
            phase (str): The phase of the evaluation. Must be one of 'TRAIN', 'VALID' or 'TEST'
        Returns:
            float: The loss of the current state.
            float: The accuracy of the current state.

        """

        if phase == "TEST":
            tproblem.test_init_op()
            msg = "TEST:"
        elif phase == "TRAIN":
            tproblem.train_eval_init_op()
            msg = "TRAIN:"
        elif phase == "VALID":
            tproblem.valid_init_op()
            msg = "VALID:"
        # evaluation loop over every batch of the corresponding evaluation set
        loss = 0.0
        accuracy = 0.0
        batchCount = 0.0
        while True:
            try:
                batch_loss, batch_accuracy = (
                    tproblem.get_batch_loss_and_accuracy()
                )
                batchCount += 1.0
                loss += batch_loss.item()
                accuracy += batch_accuracy
            except StopIteration:
                break

        loss /= batchCount
        accuracy /= batchCount

        if accuracy != 0.0:
            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss, accuracy))
        else:
            print("{0:s} loss {1:g}".format(msg, loss))

        return loss, accuracy

    def evaluate_all(
            self,
            epoch_count,
            num_epochs,
            tproblem,
            train_losses,
            valid_losses,
            test_losses,
            train_accuracies,
            valid_accuracies,
            test_accuracies,
    ):

        print("********************************")
        print(
            "Evaluating after {0:d} of {1:d} epochs...".format(
                epoch_count, num_epochs
            )
        )

        loss_, acc_ = self.evaluate(tproblem, phase="TRAIN")
        train_losses.append(loss_)
        train_accuracies.append(acc_)

        loss_, acc_ = self.evaluate(tproblem, phase="VALID")
        valid_losses.append(loss_)
        valid_accuracies.append(acc_)

        loss_, acc_ = self.evaluate(tproblem, phase="TEST")
        test_losses.append(loss_)
        test_accuracies.append(acc_)

        print("********************************")

    def evaluate_all_gan(
            self,
            epoch_count,
            num_epochs,
            tproblem,
            img_list,
            g_losses,
            d_losses,
            d_acc_real,
            d_acc_fake,
            fixed_noise,
            eval_interval,
    ):
        print("********************************")
        print(
            "Evaluating after {0:d} of {1:d} epochs...".format(
                epoch_count, num_epochs
            )
        )

        # Save generated images according to the given eval_interval
        if epoch_count % eval_interval == 0:
            with torch.no_grad():
                fake = tproblem.generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            plt.figure(figsize=(15, 15))
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            if not os.path.isdir(self._run_directory):
                os.makedirs(self._run_directory + '/images/', exist_ok=True)
            plt.savefig(self._run_directory + '/images/epoch_' + str(epoch_count) + '_' + self._file_name + '.png')

        # Create image to compare real with fake
        if epoch_count == num_epochs:
            # plot loss
            plt.figure(figsize=(15, 15))
            plt.title("Generator and Discriminator Loss ")
            plt.plot(g_losses, label="G")
            plt.plot(d_losses, label="D")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            os.makedirs(self._run_directory + '/plots/', exist_ok=True)
            plt.savefig(self._run_directory + '/plots/Loss_' + self._file_name + '.png')
            # plot accuracy
            plt.figure(figsize=(15, 15))
            plt.title("Discriminator accuracy for real and fake")
            plt.plot(d_acc_real, label="acc(D(x))")
            plt.plot(d_acc_fake, label="acc(D(G(z)))")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(self._run_directory + '/plots/Accuracy_' + self._file_name + '.png')

            # Create comparison image
            next_batch = next(iter(tproblem.data._train_eval_dataloader))
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(next_batch[0].to(config.get_default_device())[:64],
                                                     padding=5, normalize=True).cpu(), (1, 2, 0)))
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            plt.savefig(self._run_directory + '/images/compared_' + self._file_name + '.png')

        print("********************************")
        return


class StandardRunner(PTRunner):
    """A standard runner. Can run a normal training loop with fixed
    hyperparams. It should be used as a template to implement custom runners.
    """

    def __init__(self, optimizer_class, hyperparameter_names):
        super(StandardRunner, self).__init__(
            optimizer_class, hyperparameter_names
        )

    def training(
            self,
            tproblem,
            hyperparams,
            num_epochs,
            print_train_iter,
            train_log_interval,
            tb_log,
            tb_log_dir,
    ):

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    "Not possible to use tensorboard for pytorch. Reason: "
                    + e.msg,
                    RuntimeWarning,
                )
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(
                epoch_count,
                num_epochs,
                tproblem,
                train_losses,
                valid_losses,
                test_losses,
                train_accuracies,
                valid_accuracies,
                test_accuracies,
            )

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
                    batch_loss.backward()
                    opt.step()

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                    epoch_count, batch_count, batch_loss
                                )
                            )
                        if tb_log:
                            summary_writer.add_scalar(
                                "loss", batch_loss.item(), global_step
                            )

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses,
                )
                break
            else:
                continue

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "test_accuracies": test_accuracies,
        }

        return output

    def training_gan(
            self,
            tproblem,
            hyperparams,
            num_epochs,
            print_train_iter,
            train_log_interval,
            tb_log,
            tb_log_dir,
            eval_interval,
    ):
        opt_d = self._optimizer_class(tproblem.net.parameters(), **hyperparams)
        opt_g = self._optimizer_class(tproblem.generator.parameters(), **hyperparams)

        # Input vector for G
        fixed_noise = torch.randn(2048, tproblem.generator.noise_size, 1, 1, device=config.DEFAULT_DEVICE)

        # Lists to log train/test loss, accuracy and the images.
        img_list = []
        g_losses = []
        d_losses = []
        d_acc_real = []
        d_acc_fake = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    "Not possible to use tensorboard for pytorch. Reason: " + e.msg,
                    RuntimeWarning,
                )
                tb_log = False
        global_step = 0

        real_label = 1
        fake_label = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all_gan(
                epoch_count,
                num_epochs,
                tproblem,
                img_list,
                g_losses,
                d_losses,
                d_acc_real,
                d_acc_fake,
                fixed_noise,
                eval_interval,
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0

            while True:
                try:
                    # (1) Update D network
                    tproblem.net.zero_grad()
                    inputs, _ = tproblem._get_next_batch()
                    real_data = inputs.to(config.DEFAULT_DEVICE)
                    label = torch.full((len(inputs),), real_label, device=config.DEFAULT_DEVICE)
                    output = tproblem.net(real_data).view(-1)
                    loss_d_real = tproblem.loss_function(output, label)
                    loss_d_real.backward()
                    accuracy_real = output.mean().item()

                    # Train with all-fake batch
                    noise = torch.randn(len(inputs), tproblem.generator.noise_size, 1, 1, device=config.DEFAULT_DEVICE)
                    fake = tproblem.generator(noise)
                    label.fill_(fake_label)
                    output = tproblem.net(fake.detach()).view(-1)
                    loss_d_fake = tproblem.loss_function(output, label)
                    loss_d_fake.backward()
                    loss_d = loss_d_real + loss_d_fake
                    opt_d.step()

                    # (2) Update G network
                    tproblem.generator.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    output = tproblem.net(fake).view(-1)
                    loss_g = tproblem.loss_function(output, label)
                    loss_g.backward()
                    accuracy_fake = output.mean().item()
                    opt_g.step()

                    if batch_count % train_log_interval == 0:
                        g_losses.append(loss_g.item())
                        d_losses.append(loss_d.item())
                        d_acc_real.append(accuracy_real)
                        d_acc_fake.append(accuracy_fake)
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss_d {2:g}, loss_g {3:g},"
                                " accuracy_real {4:g}, accuracy_fake {5:g}".format(
                                    epoch_count, batch_count, loss_d.item(), loss_g.item(), accuracy_real, accuracy_fake
                                )
                            )
                        if tb_log:
                            summary_writer.add_scalar(
                                "loss_d", loss_d.item(), global_step
                            )
                            summary_writer.add_scalar(
                                "loss_g", loss_d.item(), global_step
                            )

                    batch_count += 1
                    global_step += 1
                except StopIteration:
                    break

            if not np.isfinite(d_losses[batch_count-1]) or not np.isfinite(g_losses[batch_count-1]):
                self._abort_routine_gan(
                    epoch_count,
                    num_epochs,
                    g_losses,
                    d_losses,
                    d_acc_real,
                    d_acc_fake,
                    loss_g,
                    loss_d,
                    accuracy_real,
                    accuracy_fake,
                )
                break
            else:
                continue

        if tb_log:
            summary_writer.close()

        output = {
            "generator_loss": g_losses,
            "discriminator_loss": d_losses,
            "discriminator_accuracy_real": d_acc_real,
            "discriminator_accuracy_fake": d_acc_fake,
        }

        return output


class LearningRateScheduleRunner(PTRunner):
    """A runner for learning rate schedules. Can run a normal training loop with fixed hyperparams or a learning rate
    schedule. It should be used as a template to implement custom runners.
    """

    def __init__(self, optimizer_class, hyperparameter_names):

        super(LearningRateScheduleRunner, self).__init__(
            optimizer_class, hyperparameter_names
        )

    def _add_training_params_to_argparse(self, parser, args, training_params):
        try:
            args["lr_sched_epochs"] = training_params["lr_sched_epochs"]
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
          decrease to 0.01*0.3=0.003' after training for 100 epochs.""",
            )

        try:
            args["lr_sched_factors"] = training_params["lr_sched_factors"]
        except KeyError:
            parser.add_argument(
                "--lr_sched_factors",
                nargs="+",
                type=float,
                help="""One or more factors (floats) by which to change the learning
          rate. The base learning rate has to be passed via '--learing_rate' and
          the epochs at which to change the learning rate have to be passed via
          '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
          --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
          then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
          decrease to 0.01*0.3=0.003' after training for 100 epochs.""",
            )

    def training(
            self,
            tproblem,
            hyperparams,
            num_epochs,
            print_train_iter,
            train_log_interval,
            tb_log,
            tb_log_dir,
            # the following are the training_params
            lr_sched_epochs=None,
            lr_sched_factors=None,
    ):
        """Performs the training and stores the metrices.

        Args:
            tproblem (deepobs.[tensorflow/pytorch].testproblems.testproblem): The testproblem instance to train on.
            hyperparams (dict): The optimizer hyperparameters to use for the training.
            num_epochs (int): The number of training epochs.
            print_train_iter (bool): Whether to print the training progress at every train_log_interval
            train_log_interval (int): Mini-batch interval for logging.
            tb_log (bool): Whether to use tensorboard logging or not
            tb_log_dir (str): The path where to save tensorboard events.
            lr_sched_epochs (list): The epochs where to adjust the learning rate.
            lr_sched_factors (list): The corresponding factors by which to adjust the learning rate.

        Returns:
            dict: The logged metrices. Is of the form: \
                {'test_losses' : [...], \
                'valid_losses': [...], \
                 'train_losses': [...],  \
                 'test_accuracies': [...], \
                 'valid_accuracies': [...], \
                 'train_accuracies': [...] \
                 } \
            where the metrices values are lists that were filled during training.
        """

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)
        if lr_sched_epochs is not None:
            lr_schedule = runner_utils.make_lr_schedule(
                optimizer=opt,
                lr_sched_epochs=lr_sched_epochs,
                lr_sched_factors=lr_sched_factors,
            )

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(
                epoch_count,
                num_epochs,
                tproblem,
                train_losses,
                valid_losses,
                test_losses,
                train_accuracies,
                valid_accuracies,
                test_accuracies,
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###
            if lr_sched_epochs is not None:
                # get the next learning rate
                lr_schedule.step(epoch_count)

                if epoch_count in lr_sched_epochs:
                    print(
                        "Setting learning rate to {0}".format(
                            lr_schedule.get_lr()
                        )
                    )

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()
                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()

                    batch_loss.backward()
                    opt.step()

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                    epoch_count, batch_count, batch_loss
                                )
                            )
                    batch_count += 1

                except StopIteration:
                    break

            # break from training if it goes wrong
            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                )
                break
            else:
                continue

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "test_accuracies": test_accuracies,
        }

        return output
