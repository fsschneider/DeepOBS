# -*- coding: utf-8 -*-

"""Helper classes and functions for the DeepOBS datasets."""

import torch
import torch.utils.data.sampler as s


class train_eval_sampler(s.Sampler):
    """A subclass of torch Sampler to easily draw the train eval set
    """
    def __init__(self, size, sub_size):
        """Args:
        size (int): The size of the original dataset.
        sub_size (int): The size of the dataset which is to be drawn from the original one."""
        self.size = size
        self.sub_size = sub_size

    def __iter__(self):
        indices = torch.randperm(self.size).tolist()
        sub_indices = indices[0:self.sub_size]
        return iter(sub_indices)

    def __len__(self):
        return self.sub_size