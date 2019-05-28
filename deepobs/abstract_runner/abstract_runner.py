# -*- coding: utf-8 -*-

"""Module implementing the abstract Runner."""
import os
import json
from .abstract_runner_utils import float2str
import time
import abc

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

    def __init__(self, optimizer_class, hyperparams):

        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._optimizer_hyperparams = hyperparams

    @abc.abstractmethod
    def run(self):
        return

    # creates the output folder structure depending on the settings of interest
    def create_output_directory(self, output_dir, output):

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
        for hp_name, hp_value in sorted(self._optimizer_hyperparams.items()):
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

        run_directory = os.path.join(output_dir, output['testproblem'], self._optimizer_name,
                                     run_folder_name)
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)

        return run_directory, file_name

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