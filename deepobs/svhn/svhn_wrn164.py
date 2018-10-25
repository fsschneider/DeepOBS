# -*- coding: utf-8 -*-
"""
This test problem is the WRN-16-4 [1] architectures on SVHN.

Training parameters according to the paper:
- bs 128
- wd 0.0005
- total training 160 epochs (~60k steps)
- Nesterov momentum with mu=0.9
- lr schedule
  - 0th epoch: 0.01
  - 80th epoch (~30k steps): 0.001
  - 120th epoch (~45k steps): 0.0001

[1]: https://arxiv.org/abs/1605.07146
"""

import svhn_wrn


def set_up(batch_size, weight_decay=0.0005):
    """Function providing the functionality for the `WideResNet`_ 16-4 architecture on `SVHN`.

    This function is a wrapper for the :class:`.svhn_wrn.set_up` to create the 16-4 variant of the WideResNet.

    The training setting in the paper were: Batch size of ``128``, weight decay of ``0.0005``, total training time of ``160`` epochs, with a learning rate schedule of ``0.01``, ``0.001`` after ``80`` epochs, ``0.0001`` after ``120`` epochs. Training was done using `Nesterov momentum` with a momentum parameter of ``0.9``.

    Args:
        batch_size (int): Batch size of the data points.
        weight_decay (float): Weight decay factor. Defaults to ``0.0005``.

    Returns:
        svhn_wrn.set_up: Setup class for WideResNets on `SVHN`, :class:`.svhn_wrn.set_up`.

    .. _WideResNet: https://arxiv.org/abs/1605.07146
    """
    return svhn_wrn.set_up(batch_size, num_residual_units=2, k=4, weight_decay=weight_decay, bn_decay=0.9)
