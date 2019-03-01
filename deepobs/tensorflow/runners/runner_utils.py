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


def make_lr_schedule(lr_base, lr_sched_epochs, lr_sched_factors):
    """Creates a learning rate schedule in the form of a dictionary.

  After ``lr_sched_epochs[i]`` epochs of training, the learning rate will be set
  to ``lr_sched_factors[i] * lr_base``. The schedule is given as a dictionary
  mapping epoch number to learning rate. The learning rate for epoch 0 (that is
  ``lr_base``) will automatically be added to the schedule.

  Examples:
    - ``make_schedule(0.3, [50, 100], [0.1, 0.01])`` yields
      ``{0: 0.3, 50: 0.03, 100: 0.003}``.
    - ``make_schedule(0.3)`` yields ``{0: 0.3}``.
    - ``make_schedule(0.3, [], [])`` yields ``{0: 0.3}``.

  Args:
    lr_base: A base learning rate (float).
    lr_sched_epochs: A (possibly empty) list of integers, specifying epochs at
        which to decrease the learning rate.
    lr_sched_factors: A (possibly empty) list of floats, specifying factors by
        which to decrease the learning rate.

  Returns:
    sched: A dictionary mapping epoch numbers to learning rates.
  """

    if lr_sched_epochs is None and lr_sched_factors is None:
        return {0: lr_base}

    # Make sure learning rate schedule has been properly specified
    if lr_sched_epochs is None or lr_sched_factors is None:
        raise TypeError(
            """Specifiy *both* lr_sched_epochs and lr_sched_factors.""")
    if ((not isinstance(lr_sched_epochs, list))
            or (not isinstance(lr_sched_factors, list))
            or (len(lr_sched_epochs) != len(lr_sched_factors))):
        raise ValueError(
            """lr_sched_epochs and lr_sched_factors must be lists of
                     the same length.""")

    # Create schedule as dictionary epoch->factor; add value for epoch 0.
    sched = {n: f * lr_base for n, f in zip(lr_sched_epochs, lr_sched_factors)}
    sched[0] = lr_base
    return sched
