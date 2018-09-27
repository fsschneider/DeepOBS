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

import _cifar100_wrn


def set_up(batch_size, weight_decay=0.0005):
    return _cifar100_wrn.set_up(batch_size, num_residual_units=6, k=4,
                                weight_decay=weight_decay, bn_decay=0.9)
