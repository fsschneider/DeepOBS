# -*- coding: utf-8 -*-
"""MNIST DeepOBS dataset."""

from __future__ import print_function

from . import dataset
from .. import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms
from .datasets_utils import train_eval_sampler

class fmnist(dataset.DataSet):
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
                 data_augmentation = False,
                 train_eval_size=10000):
        """Creates a new MNIST instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``60 000`` for train, ``10 000``
          for test) the remainder is dropped in each epoch (after shuffling).
      train_eval_size (int): Size of the train eval data set.
          Defaults to ``10 000`` the size of the test set.
    """
        self._name = "fmnist"
        self._data_augmentation = data_augmentation
        self._train_eval_size = train_eval_size
        super(fmnist, self).__init__(batch_size)

    def _make_dataloader(self, train = True, shuffle=True, data_augmentation = False, sampler=None):
        transform = transforms.ToTensor()
        dataset = datasets.FashionMNIST(root = config.get_data_dir(), train = train, download = True, transform = transform)
        loader = dat.DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, drop_last=True, pin_memory=self._pin_memory, num_workers=1, sampler=sampler)
        return loader

    def _make_train_dataloader(self):
        return self._make_dataloader(train=True, shuffle = True, data_augmentation = self._data_augmentation, sampler=None)

    def _make_test_dataloader(self):
        return self._make_dataloader(train=False, shuffle = False, data_augmentation = False, sampler=None)

    def _make_train_eval_dataloader(self):
        size = len(self._train_dataloader.dataset)
        sampler = train_eval_sampler(size, self._train_eval_size)
        return self._make_dataloader(train=True, shuffle=False, data_augmentation=False, sampler=sampler)