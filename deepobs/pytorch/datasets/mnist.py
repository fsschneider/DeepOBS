# -*- coding: utf-8 -*-
"""MNIST DeepOBS dataset."""

from __future__ import print_function

from . import dataset
from .. import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms
from .datasets_utils import train_eval_sampler

class mnist(dataset.DataSet):
    """DeepOBS data set class for the `MNIST\
    <http://yann.lecun.com/exdb/mnist/>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``60 000`` for train, ``10 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    train_eval_size (int): Size of the train eval data set.
        Defaults to ``10 000`` the size of the test set.

  Methods:
      _make_dataloader: A helper that is shared by all three data loader methods.
  """

    def __init__(self,
                 batch_size,
                 train_eval_size=10000):
        """Creates a new MNIST instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``60 000`` for train, ``10 000``
          for test) the remainder is dropped in each epoch (after shuffling).
      train_eval_size (int): Size of the train eval data set.
          Defaults to ``10 000`` the size of the test set.
    """
        self._name = "mnist"
        self._train_eval_size = train_eval_size
        super(mnist, self).__init__(batch_size)

    def _make_dataloader(self, train, shuffle = True, sampler = None):
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root = config.get_data_dir(), train = train, download = True, transform = transform)
        loader = dat.DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, sampler=sampler, drop_last=True, pin_memory=True, num_workers=4)
        return loader

    def _make_train_dataloader(self):
        return self._make_dataloader(train=True, shuffle = False)

    def _make_test_dataloader(self):
        return self._make_dataloader(train=False, shuffle = False)

    def _make_train_eval_dataloader(self):
        size = len(self._train_dataloader.dataset)
        sampler = train_eval_sampler(size, self._train_eval_size)
        return self._make_dataloader(train=True, shuffle=False, sampler = sampler)