# -*- coding: utf-8 -*-
"""Utility functions for running optimizers."""

from torch.optim import lr_scheduler
import numpy as np

def make_lr_schedule(optimizer, lr_sched_epochs = None, lr_sched_factors = None):
    """Creates a learning rate schedule in the form of a torch.optim.lr_scheduler.LambdaLR instance.

  After ``lr_sched_epochs[i]`` epochs of training, the learning rate will be set
  to ``lr_sched_factors[i] * lr_base``.

  Examples:
    - ``make_schedule(optim.SGD(net.parameters(), lr = 0.5), [50, 100], [0.1, 0.01])`` yields
      to the following schedule for the SGD optimizer on the parameters of net:
      SGD uses lr = 0.5 for epochs 0 to 49.
      SGD uses lr = 0.5*0.1 = 0.05 for epochs 50 to 99.
      SGD uses lr = 0.5*0.01 = 0.005 for epochs 100 to end.

  Args:
    optimizer: The optimizer for which the schedule is set. It already holds the base learning rate.
    lr_sched_epochs: A list of integers, specifying epochs at
        which to decrease the learning rate.
    lr_sched_factors: A list of floats, specifying factors by
        which to decrease the learning rate.

  Returns:
    sched: A torch.optim.lr_scheduler.LambdaLR instance with a function that determines the learning rate at every epoch.
  """

    if (lr_sched_factors is None) or (lr_sched_epochs is None):
        determine_lr = lambda epoch: 1
    else:
        def determine_lr(epoch):
            if epoch < lr_sched_epochs[0]:
                return 1
            else:
                help_array = np.array(lr_sched_epochs)
                index = np.argmax(np.where(help_array <= epoch)[0])
                return lr_sched_factors[index]

    sched = lr_scheduler.LambdaLR(optimizer, determine_lr)
    return sched
