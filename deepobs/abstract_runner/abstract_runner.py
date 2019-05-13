# -*- coding: utf-8 -*-

"""Module implementing the abstract Runner."""
import os
import json
from .abstract_runner_utils import float2str
import time
import abc

class Runner(abc.ABC):
    """Captures everything that is common to both frameworks and every runner type.
    This includes folder creation amd writing of the output to the folder"""

    def __init__(self, optimizer_class, hyperparams):

        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._optimizer_hyperparams = hyperparams

    @abc.abstractmethod
    def run(self):
        pass

    # creates the output folder structure depending on the settings of interest
    # TODO this would also become way easier if I wrapp the 'importance settings dict'
    def create_output_directory(self,
                                testproblem,
                                num_epochs,
                                batch_size,
                                weight_decay,
                                random_seed,
                                output_dir,
                                **training_params):

        run_folder_name = "num_epochs__" + str(
        num_epochs) + "__batch_size__" + str(batch_size) + "__"
        if weight_decay is not None:
            run_folder_name += "weight_decay__{0:s}".format(
                float2str(weight_decay))

        # Add all hyperparameters to the name (sorted alphabetically).
        # TODO what happens if a value is neither string nor float?
        for hp_name, hp_value in sorted(self._optimizer_hyperparams.items()):
            run_folder_name += "__{0:s}".format(hp_name)
            run_folder_name += "__{0:s}".format(
                float2str(hp_value) if isinstance(hp_value, float)
                                    else str(hp_value)
                                    )

        # Add training parameters to the name (sorted alphabetically)
        # TODO what happens if a value is neither string nor float?
        for tp_name, tp_value in sorted(training_params.items()):
            run_folder_name += "__{0:s}".format(tp_name)
            run_folder_name += "__{0:s}".format(
                float2str(hp_value) if isinstance(tp_value, float)
                                    else str(tp_value)
                                    )

        file_name = "random_seed__{0:d}__".format(random_seed)
        file_name += time.strftime("%Y-%m-%d-%H-%M-%S")

        run_directory = os.path.join(output_dir, testproblem, self._optimizer_name,
                                     run_folder_name)
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)

        return run_directory, file_name

    @staticmethod
    def write_output(output, run_folder_name, file_name):
        # Dump output into json file.
        with open(os.path.join(run_folder_name, file_name + ".json"), "w") as f:
                json.dump(output, f)