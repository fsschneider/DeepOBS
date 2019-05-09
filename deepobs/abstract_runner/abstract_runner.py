# -*- coding: utf-8 -*-

"""Module implementing the abstract Runner."""
import os
import json
from . import abstract_runner_utils

class Runner(object):
    """Captures everything that is common to both frameworks and every runner type.
    This includes folder creation amd writing of the output to the folder"""

    def __init__(self, optimizer_class, hyperparams):

        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._hyperparams = hyperparams

    def run(self):
        raise NotImplementedError(
            """'Runner' is an abstract base class, please use
        one of the sub-classes.""")

    # creates the output folder structure depending on the settings of interest
    def create_output_folder(self,
                             hyperparams,
                             testproblem,
                             output_dir,
                             weight_decay,
                             batch_size,
                             num_epochs,
                             learning_rate,
                             lr_sched_epochs,
                             lr_sched_factors,
                             random_seed):

        run_folder_name, file_name = abstract_runner_utils.make_run_name(
            weight_decay, batch_size, num_epochs, learning_rate,
            lr_sched_epochs, lr_sched_factors, random_seed,
            **hyperparams)
        directory = os.path.join(output_dir, testproblem, self._optimizer_name,
                                 run_folder_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory, file_name

    # writes the output, given a dictionary determined by the individual runner.
    def write_output(self, output, directory, file_name):
        # Dump output into json file.
        with open(os.path.join(directory, file_name + ".json"), "w") as f:
                json.dump(output, f)