# -*- coding: utf-8 -*-
"""Utility functions for running optimizers."""

import time

def float2str(x):
    s = "{:.10e}".format(x)
    mantissa, exponent = s.split("e")
    return mantissa.rstrip("0") + "e" + exponent


def make_run_name(weight_decay, batch_size, num_epochs, learning_rate,
                  lr_sched_epochs, lr_sched_factors, random_seed,
                  **optimizer_hyperparams):
    """Creates a name for the output file of an optimizer run.

  Args:
    weight_decay (float): The weight decay factor used (or ``None`` to signify
        the testproblem's default).
    batch_size (int): The mini-batch size used.
    num_epochs (int): The number of epochs trained.
    learning_rate (float): The learning rate used.
    lr_sched_epochs (list): A list of epoch numbers (positive integers) that
        mark learning rate changes.
    lr_sched_factors (list): A list of factors (floats) by which to change the
        learning rate.
    random_seed (int): Random seed used.

  Returns:
    run_folder_name: Name for the run folder consisting of num_epochs,
        batch_size, weight_decay, all the optimizer hyperparameters, and the
        learning rate (schedule).
    file_name: Name for the output file, consisting of random seed and a time
        stamp.
  """
    run_folder_name = "num_epochs__" + str(
        num_epochs) + "__batch_size__" + str(batch_size) + "__"
    if weight_decay is not None:
        run_folder_name += "weight_decay__{0:s}__".format(
            float2str(weight_decay))

    # Add all hyperparameters to the name (sorted alphabetically).
    for hp_name, hp_value in sorted(optimizer_hyperparams.items()):
        run_folder_name += "{0:s}__".format(hp_name)
        run_folder_name += "{0:s}__".format(
            float2str(hp_value) if isinstance(hp_value, float
                                              ) else str(hp_value))
    if lr_sched_epochs is None:
        run_folder_name += "lr__{0:s}".format(float2str(learning_rate))
    else:
        run_folder_name += ("lr_schedule__{0:d}_{1:s}".format(
            0, float2str(learning_rate)))
        for epoch, factor in zip(lr_sched_epochs, lr_sched_factors):
            run_folder_name += ("_{0:d}_{1:s}".format(
                epoch, float2str(factor * learning_rate)))
    file_name = "random_seed__{0:d}__".format(random_seed)
    file_name += time.strftime("%Y-%m-%d-%H-%M-%S")
    return run_folder_name, file_name


