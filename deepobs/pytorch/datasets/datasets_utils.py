# -*- coding: utf-8 -*-
from torch.utils.data import sampler
import numpy as np

class train_eval_sampler(sampler.Sampler):
    """A subclass of torch Sampler to easily draw the train eval set
    """
    def __init__(self, size, sub_size):
        self.size = size
        self.sub_size = sub_size
    def __iter__(self):
        indices = np.arange(self.size)
        return iter(np.random.choice(indices, size = self.sub_size, replace = False).tolist())
    def __len__(self):
        return self.sub_size