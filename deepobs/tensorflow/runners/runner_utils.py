# -*- coding: utf-8 -*-
"""Utility functions for running optimizers."""


def float2str(x):
    s = "{:.10e}".format(x)
    mantissa, exponent = s.split("e")
    return mantissa.rstrip("0") + "e" + exponent


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
