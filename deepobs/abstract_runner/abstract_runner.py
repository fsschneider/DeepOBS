# -*- coding: utf-8 -*-

"""Module implementing the abstract Runner."""
import os
import json
from .abstract_runner_utils import float2str
from .abstract_runner_utils import _add_hp_to_argparse
import time
import abc
import argparse
import warnings


class Runner(abc.ABC):
    """Abstract base class for all different runners in DeepOBS.
    Captures everything that is common to both frameworks and every runner type.
    This includes folder creation amd writing of the output to the folder.

    Attributes:
        _optimizer_class: See argument optimizer_class
        _optimizer_name: The name of the optimizer class
        _hyperparameter_names: A nested dictionary that lists all hyperparameters of the optimizer,
        their type and their default values

    Methods:
        run: An abstract method that is overwritten by the tensorflow and pytorch
        specific subclasses. It performs the actual run on a testproblem.
        training: An abstract method that performs the actual training and is overwritten by the subclasses.
        create_output_directory: Creates the output folder of the run.
        write_output: Writes the output of the run to the output directory.
    """

    def __init__(self, optimizer_class, hyperparameter_names):
        """
        Args:
            optimizer_class: The optimizer class of the optimizer that is run on
            the testproblems. For PyTorch this must be a subclass of torch.optim.Optimizer. For
            TensorFlow a subclass of tf.train.Optimizer.

            hyperparameter_names (dict): A nested dictionary that lists all hyperparameters of the optimizer,
            their type and their default values (if they have any) in the form: {'<name>': {'type': <type>, 'default': <default value>}},
            e.g. for torch.optim.SGD with momentum:
            {'lr': {'type': float},
            'momentum': {'type': float, 'default': 0.99},
            'uses_nesterov': {'type': bool, 'default': False}}
        """
        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._hyperparameter_names = hyperparameter_names

    def run(self,
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
            **training_params
            ):

        """Runs a testproblem with the optimizer_class. Has the following tasks:
            1. setup testproblem
            2. run the training (must be implemented by subclass)
            3. merge and write output

        Args:
            testproblem (str): Name of the testproblem.
            hyperparams (dict): The explizit values of the hyperparameters of the optimizer that are used for training
            batch_size (int): Mini-batch size for the training data.
            num_epochs (int): The number of training epochs.
            random_seed (int): The torch random seed.
            data_dir (str): The path where the data is stored.
            output_dir (str): Path of the folder where the results are written to.
            weight_decay (float): Regularization factor for the testproblem.
            no_logs (bool): Whether to write the output or not.
            train_log_interval (int): Mini-batch interval for logging.
            print_train_iter (bool): Whether to print the training progress at each train_log_interval.
            tb_log (bool): Whether to use tensorboard logging or not
            tb_log_dir (str): The path where to save tensorboard events.
            training_params (dict): Kwargs for the training method.

        Returns:
            dict: {<...meta data...>
                'test_losses' : test_losses
                 'train_losses': train_losses,
                 'test_accuracies': test_accuracies,
                 'train_accuracies': train_accuracies,
                 'analyzable_training_params': {...}
                 }
            were <...meta data...> stores the run args.

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
            train_log_interval,
            print_train_iter,
            tb_log,
            tb_log_dir,
            training_params)

        output = self._run(**args)
        return output

    @abc.abstractmethod
    def _run(self,
            testproblem,
            hyperparams,
            batch_size,
            num_epochs,
            random_seed,
            data_dir,
            output_dir,
            weight_decay,
            no_logs,
            train_log_interval,
            print_train_iter,
            tb_log,
            tb_log_dir,
            **training_params):
        return

    @abc.abstractmethod
    def training(self, tproblem, hyperparams, num_epochs, print_train_iter, train_log_interval, tb_log, tb_log_dir, **training_params):
        """Performs the training and stores the metrices.
            Args:
                tproblem (deepobs.[tensorflow/pytorch].testproblems.testproblem): The testproblem instance to train on.
                hyperparams (dict): The optimizer hyperparameters to use for the training.
                num_epochs (int): The number of training epochs.
                print_train_iter (bool): Whether to print the training progress at every train_log_interval
                train_log_interval (int): Mini-batch interval for logging.
                tb_log (bool): Whether to use tensorboard logging or not
                tb_log_dir (str): The path where to save tensorboard events.
                **training_params (dict): Kwargs for additional training parameters that are implemented by subclass.

            Returns:
                dict: The logged metrices. Is of the form:
                    ```{'test_losses' : [...]
                     'train_losses': [...],
                     'test_accuracies': [...],
                     'train_accuracies': [...]
                     }```
            where the metrices values are lists that were filled during training.
        """
        return

    @staticmethod
    @abc.abstractmethod
    def evaluate(*args, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def create_testproblem(*args, **kwargs):
        pass

    def _add_training_params_to_argparse(self, parser, args, training_params):
        """Overwrite this method to specify how your runner should read in additional training_parameters and to add them to argparse"""
        pass

    def _add_hyperparams_to_argparse(self, parser, args, hyperparams):
        """Overwrite this method to specify how your runner should read in optimizer hyper_parameters and to add them to argparse"""
        if hyperparams is None:    # if no hyperparams dict is passed to run()
            for hp_name, hp_specification in self._hyperparameter_names.items():
                _add_hp_to_argparse(parser, self._optimizer_name, hp_specification, hp_name)

        else:     # if there is one, fill the missing params from command line
            for hp_name, hp_specification in self._hyperparameter_names.items():
                if hp_name in hyperparams:
                    args[hp_name] = hyperparams[hp_name]
                else:
                    _add_hp_to_argparse(parser, self._optimizer_name, hp_specification, hp_name)

    def _add_training_params_to_output_dir_name(self, output, run_folder_name):
        """Overwrite this method to specify how your runner should format additional training_parameters in the run folder name."""
        for tp_name, tp_value in sorted(output['training_params'].items()):
            if tp_value is not None:
                run_folder_name += "__{0:s}".format(tp_name)
                run_folder_name += "__{0:s}".format(
                    float2str(tp_value) if isinstance(tp_value, float) else str(tp_value))
        return run_folder_name

    def _add_hyperparams_to_output_dir_name(self, output, run_folder_name):
        """Overwrite this method to specify how your runner should format optimizer hyper_parameters in the run folder name."""
        for hp_name, hp_value in sorted(output['optimizer_hyperparams'].items()):
            run_folder_name += "__{0:s}".format(hp_name)
            run_folder_name += "__{0:s}".format(
                float2str(hp_value) if isinstance(hp_value, float) else str(hp_value))
        return run_folder_name

    def parse_args(self,
            testproblem,
            hyperparams,
            batch_size,
            num_epochs,
            random_seed,
            data_dir,
            output_dir,
            weight_decay,
            no_logs,
            train_log_interval,
            print_train_iter,
            tb_log,
            tb_log_dir,
            training_params):

        """Constructs an argparse.ArgumentParser and parses the arguments from command line.
        Args:
                testproblem (str): Name of the testproblem.
                hyperparams (dict): The explizit values of the hyperparameters of the optimizer that are used for training
                batch_size (int): Mini-batch size for the training data.
                num_epochs (int): The number of training epochs.
                random_seed (int): The torch random seed.
                data_dir (str): The path where the data is stored.
                output_dir (str): Path of the folder where the results are written to.
                weight_decay (float): Regularization factor for the testproblem.
                no_logs (bool): Whether to write the output or not.
                train_log_interval (int): Mini-batch interval for logging.
                print_train_iter (bool): Whether to print the training progress at each train_log_interval.
                tb_log (bool): Whether to use tensorboard logging or not
                tb_log_dir (str): The path where to save tensorboard events.
                training_params (dict): Kwargs for the training method.

        Returns: args (dict): A dicionary of all arguments.
            """
        args = {}
        parser = argparse.ArgumentParser(description='Arguments for running optimizer script.')

        if testproblem is None:
            parser.add_argument('testproblem')
        else:
            args['testproblem'] = testproblem

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
            args['weight_decay'] = weight_decay

        if batch_size is None:
            parser.add_argument(
                "--batch_size",
                "--bs",
                type=int,
                help="The batch size (positive integer).")
        else:
            args['batch_size'] = batch_size

        if num_epochs is None:
            parser.add_argument(
                "-N",
                "--num_epochs",
                type=int,
                help="Total number of training epochs.")
        else:
            args['num_epochs'] = num_epochs

        if random_seed is None:
            parser.add_argument(
                "-r",
                "--random_seed",
                type=int,
                default=42,
                help="An integer to set as tensorflow's random seed.")
        else:
            args['random_seed'] = random_seed

        if data_dir is None:
            parser.add_argument(
                "--data_dir",
                help="""Path to the base data dir. If
      not specified, DeepOBS uses its default.""")
        else:
            args['data_dir'] = data_dir

        if output_dir is None:
            parser.add_argument(
                "--output_dir",
                type=str,
                default="./results",
                help="""Path to the base directory in which output files will be
          stored. Results will automatically be sorted into subdirectories of
          the form 'testproblem/optimizer'.""")
        else:
            args['output_dir'] = output_dir

        if no_logs is None:
            parser.add_argument(
                "--no_logs",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to not save any json logging files.""")
        else:
            args['no_logs'] = no_logs

        if train_log_interval is None:
            parser.add_argument(
                "--train_log_interval",
                type = int,
                default=10,
                help="""Interval of steps at which to log training loss.""")
        else:
            args['train_log_interval'] = train_log_interval

        if print_train_iter is None:
            parser.add_argument(
                "--print_train_iter",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to print the mini-batch-loss at the train_log_interval.""")
        else:
            args['print_train_iter'] = print_train_iter

        if tb_log is None:
            parser.add_argument(
                "--tb_log",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to save tensorboard logging files.""")
        else:
            args['tb_log'] = tb_log

        if tb_log_dir is None:
            parser.add_argument(
                "--tb_log_dir",
                type=str,
                default="./tb_log",
                help="""Path to the directory where the tensorboard logs are saved.""")
        else:
            args['tb_log_dir'] = tb_log_dir

        # add hyperparams and training params
        self._add_hyperparams_to_argparse(parser, args, hyperparams)
        self._add_training_params_to_argparse(parser, args, training_params)

        cmdline_args = vars(parser.parse_args())
        args.update(cmdline_args)

        # put all optimizer hyperparams in one subdict
        args['hyperparams'] = {}
        for hp in self._hyperparameter_names:
            args['hyperparams'][hp] = args[hp]
            del args[hp]

        return args

    def create_output_directory(self, output_dir, output):
        """Creates the output directory of the run.
        Args:
            output_dir (str): The path to the results folder
            output (dict): A dict than contains the metrices and main settings
            from the training run and a subdict called 'analyzable_training_params'
            that holds additional training_params that need to be analyzed.
        Returns:
            run_directory (str): Path to the run directory which is named
            after all relevant settings.
            file_name (str): JSON file name of the run that is named after the
            seed and terminating time of the run.
        """

        # add everything mandatory to the name
        run_folder_name = "num_epochs__" + str(
            output['num_epochs']) + "__batch_size__" + str(output['batch_size'])
        if output['weight_decay'] is not None:
            run_folder_name += "__weight_decay__{0:s}".format(
                float2str(output['weight_decay']))

        # Add all hyperparameters to the name.
        run_folder_name = self._add_hyperparams_to_output_dir_name(output, run_folder_name)

        # Add training parameters to the name.
        run_folder_name = self._add_training_params_to_output_dir_name(output, run_folder_name)

        file_name = "random_seed__{0:d}__".format(output['random_seed'])
        file_name += time.strftime("%Y-%m-%d-%H-%M-%S")

        run_directory = os.path.join(output_dir, output['testproblem'], output['optimizer_name'],
                                     run_folder_name)
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)

        return run_directory, file_name
    
    def _post_process_output(self, output, testproblem, batch_size, num_epochs, random_seed, weight_decay, hyperparams, **training_params):
        """Ensures that for both frameworks the structure of the output is the same"""
        
        # remove test accuracy if it is not available
        if 'test_accuracies' in output:
            if all(output['test_accuracies']) == 0:
                del output['test_accuracies']
                del output['train_accuracies']
        
        # merge meta data to output dict
        output = {'testproblem': testproblem,
                  'batch_size': batch_size,
                  'num_epochs': num_epochs,
                  'random_seed': random_seed,
                  'weight_decay': weight_decay,
                  'optimizer_name': self._optimizer_name,
                  'optimizer_hyperparams': hyperparams,
                  'training_params': training_params,
                  **output}
        
        return output
            
    @staticmethod
    def write_output(output, run_folder_name, file_name):
        """Writes the JSON output.
        Args:
            output (dict): Output of the training loop of the runner.
            run_folder_name (str): The name of the output folder.
            file_name (str): The file name where the output is written to.
        """
        with open(os.path.join(run_folder_name, file_name + ".json"), "w") as f:
            json.dump(output, f, indent=4)

    @staticmethod
    def _abort_routine(epoch_count, num_epochs, train_losses, test_losses, train_accuracies, test_accuracies,
                       minibatch_train_losses):
        """A routine that is executed if a training run is aborted (loss is NaN or Inf)."""

        warnings.warn('Breaking from run after epoch ' + str(epoch_count) + ' due to wrongly calibrated optimization (Loss is Nan or Inf). The metrices for the remaining epochs will be filled with the initial performance values.', RuntimeWarning)

        # fill the rest of the metrices with initial observations
        for i in range(epoch_count, num_epochs):
            train_losses.append(train_losses[0])
            test_losses.append(test_losses[0])
            train_accuracies.append(train_accuracies[0])
            test_accuracies.append(test_accuracies[0])
            minibatch_train_losses.append(minibatch_train_losses[0])
        return train_losses, test_losses, train_accuracies, test_accuracies, minibatch_train_losses


class PTRunner(Runner):
    """The abstract class for runner in the pytorch framework."""

    def __init__(self, optimizer_class, hyperparameter_names):
        super(PTRunner, self).__init__(optimizer_class, hyperparameter_names)

    @abc.abstractmethod
    def training(self, tproblem, hyperparams, num_epochs, print_train_iter, train_log_interval, tb_log, tb_log_dir,
                 **training_params):
        return

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
    def create_testproblem(testproblem, batch_size, weight_decay, random_seed):
        """Sets up the deepobs.pytorch.testproblems.testproblem instance.
        Args:
            testproblem (str): The name of the testproblem.
            batch_size (int): Batch size that is used for training
            weight_decay (float): Regularization factor
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
            If false, it is evaluated on the train evaluation set.
        Returns:
            float: The loss of the current state.
            float: The accuracy of the current state.
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

        if accuracy != 0.0:
            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss, accuracy))
        else:
            print("{0:s} loss {1:g}".format(msg, loss))

        return loss, accuracy


class TFRunner(Runner):
    def __init__(self, optimizer_class, hyperparameter_names):

        super(TFRunner, self).__init__(optimizer_class, hyperparameter_names)

    def _run(self,
             testproblem=None,
             hyperparams=None,
             batch_size=None,
             num_epochs=None,
             random_seed=None,
             data_dir=None,
             output_dir=None,
             weight_decay=None,
             no_logs=None,
             train_log_interval=None,
             print_train_iter=None,
             tb_log=None,
             tb_log_dir=None,
             **training_params):

        if batch_size is None:
            batch_size = global_config.get_testproblem_default_setting(testproblem)['batch_size']
        if num_epochs is None:
            num_epochs = global_config.get_testproblem_default_setting(testproblem)['num_epochs']

        if data_dir is not None:
            config.set_data_dir(data_dir)

        tproblem = self.create_testproblem(testproblem, batch_size, weight_decay, random_seed)

        output = self.training(tproblem, hyperparams, num_epochs, print_train_iter, train_log_interval, tb_log,
                               tb_log_dir, **training_params)
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
    def training(self, tproblem, hyperparams, num_epochs, print_train_iter, train_log_interval, tb_log, tb_log_dir,
                 **training_params):
        """asas"""
        return


