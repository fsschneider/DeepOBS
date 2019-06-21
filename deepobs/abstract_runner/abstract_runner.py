# -*- coding: utf-8 -*-

"""Module implementing the abstract Runner."""
import os
import json
from .abstract_runner_utils import float2str
from .abstract_runner_utils import StoreDictKeyPair
import time
import abc
import argparse

class Runner(abc.ABC):
    """Abstract base class for all different runners in DeepOBS.
    Captures everything that is common to both frameworks and every runner type.
    This includes folder creation amd writing of the output to the folder.

    Args:
    optimizer_class: The optimizer class of the optimizer that is run on
    the testproblems. For tensorflow this is a subclass of tf.train.Optimizer.
    For pytorch this is a subclass of torch.optim.Optimizer

    hyperparams (dict): A dict containing the hyperparams for the optimizer_class.

    Attributes:
    _optimizer_class: See argument optimizer_class
    _optimizer_name: The name of the optimizer class
    _optimizer_hyperparams: See argument hyperparams

    Methods:
    run: An abstract method that is overwritten by the tensorflow and pytorch
    specific subclasses. It performs the actual run on a testproblem.

    create_output_directory: Creates the output folder of the run.

    write_output: Writes the output of the run to the output directory.
    """

    def __init__(self, optimizer_class):

        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__

# TODO train log interval and tf logging?
    @abc.abstractmethod
    def run(self,
            testproblem,
            hyperparams,
            batch_size = None,
            num_epochs = None,
            random_seed=42,
            data_dir=None,
            output_dir='./results',
            weight_decay=None,
            no_logs=False,
            **training_params):
        return

    @staticmethod
    def parse_args(testproblem,
            hyperparams,
            batch_size,
            num_epochs,
            random_seed,
            data_dir,
            output_dir,
            weight_decay,
            no_logs,
            **training_params):

# TODO train log interval and tf logging?
        args = {}
        parser = argparse.ArgumentParser(description='Arguments for running optimizer script.')

        if testproblem is None:
            parser.add_argument('testproblem')
        else:
            args['testproblem'] = testproblem

        if hyperparams is None:
            parser.add_argument('hyperparams', action = StoreDictKeyPair)
        else:
            args['hyperparams'] = hyperparams

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
                default="results",
                help="""Path to the base directory in which output files will be
          stored. Results will automatically be sorted into subdirectories of
          the form 'testproblem/optimizer'.""")
        else:
            args['output_dir'] = output_dir

        if not training_params:
            parser.add_argument(
                "--training_params",
                help="""Additional training parameters as key-value pairs.""",
                action = StoreDictKeyPair,
                default = {})
        else:
            args['training_params'] = training_params

        if no_logs is None:
            parser.add_argument(
                "--no_logs",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to not save any json logging files.""")
        else:
            args['no_logs'] = no_logs

        cmdline_args = vars(parser.parse_args())
        args.update(cmdline_args)
        return args

    @staticmethod
    def create_output_directory(output_dir, output):

        """Creates the output directory of the run.
        Input:
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

        # Add all hyperparameters to the name (sorted alphabetically).
        for hp_name, hp_value in sorted(output['optimizer_hyperparams'].items()):
            run_folder_name += "__{0:s}".format(hp_name)
            run_folder_name += "__{0:s}".format(
                float2str(hp_value) if isinstance(hp_value, float)
                                    else str(hp_value)
                                    )

        # Add analyzable training parameters to the name (sorted alphabetically)
        for tp_name, tp_value in sorted(output['analyzable_training_params'].items()):
            if tp_value is not None:
                run_folder_name += "__{0:s}".format(tp_name)
                run_folder_name += "__{0:s}".format(
                    float2str(hp_value) if isinstance(tp_value, float)
                                        else str(tp_value)
                                        )

        file_name = "random_seed__{0:d}__".format(output['random_seed'])
        file_name += time.strftime("%Y-%m-%d-%H-%M-%S")

        run_directory = os.path.join(output_dir, output['testproblem'], output['optimizer_name'],
                                     run_folder_name)
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)

        return run_directory, file_name
    
    def _post_process_output(self, output, testproblem, batch_size, num_epochs, random_seed, weight_decay, hyperparams):
        '''Ensures that for both frameworks the structure of the output is the same'''
        
        # remove test accuracy if it is not available
        if output['test_accuracies']:
            if all(output['test_accuracies']) == 0:
                del output['test_accuracies']
                del output['train_accuracies']
        
        # add empty analyzable trainig_params dict if the user forgot it
        if not output['analyzable_training_params']:
            output['analyzable_training_params'] = {}
        
        # merge meta data to output dict
        # TODO Attention! train_params that default are not written to output (e.g. train log interval)!
        output = {'testproblem': testproblem,
                  'batch_size': batch_size,
                  'num_epochs': num_epochs,
                  'random_seed': random_seed,
                  'weight_decay': weight_decay,
                  'optimizer_name': self._optimizer_name,
                  'optimizer_hyperparams': hyperparams,
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
                json.dump(output, f)