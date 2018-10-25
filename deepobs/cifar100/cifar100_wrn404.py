# -*- coding: utf-8 -*-
"""
This test problem is the WRN-40-4 [1] architectures on CIFAR-100.

Training parameters according to the paper:
- bs 128
- wd 0.0005
- total training 200 epochs (80,000 steps)
- Nesterov momentum with mu=0.9
- lr schedule
  - 0th epoch: 0.1
  - 60th epoch (~24k steps): 0.02
  - 120th epoch (~48k steps): 0.004
  - 160th epoch (~64k steps): 0.0008

[1]: https://arxiv.org/abs/1605.07146
"""

import cifar100_wrn


def set_up(batch_size, weight_decay=0.0005):
    """Function providing the functionality for the `WideResNet`_ 40-4 architecture on `CIFAR-100`.

    This function is a wrapper for the :class:`.cifar100_wrn.set_up` to create the 40-4 variant of the WideResNet.

    The training setting in the paper were: Batch size of ``128``, weight decay of ``0.0005``, total training time of ``200`` epochs, with a learning rate schedule of ``0.1``, ``0.02`` after ``60`` epochs, ``0.004`` after ``120`` epochs and ``0.0008`` after ``160`` epochs. Training was done using `Nesterov momentum` with a momentum parameter of ``0.9``.

    Args:
        batch_size (int): Batch size of the data points.
        weight_decay (float): Weight decay factor. Defaults to ``0.0005``.

    Returns:
        cifar100_wrn.set_up: Setup class for WideResNets on `CIFAR-100`, :class:`.cifar100_wrn.set_up`.

    .. _WideResNet: https://arxiv.org/abs/1605.07146
    """
    return cifar100_wrn.set_up(batch_size, num_residual_units=6, k=4, weight_decay=weight_decay, bn_decay=0.9)
