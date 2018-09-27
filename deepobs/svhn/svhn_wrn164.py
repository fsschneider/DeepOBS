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

import _svhn_wrn


def set_up(batch_size, weight_decay=0.0005):
    return _svhn_wrn.set_up(batch_size, num_residual_units=2, k=4,
                            weight_decay=weight_decay, bn_decay=0.9)
