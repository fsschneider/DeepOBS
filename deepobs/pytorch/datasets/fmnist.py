# -*- coding: utf-8 -*-
"""Fashion-MNIST DeepOBS dataset."""

from __future__ import print_function
from . import dataset
from deepobs import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms
from .datasets_utils import train_eval_sampler


class fmnist(dataset.DataSet):
    """DeepOBS data set class for the `Fashion-MNIST (FMNIST)\
    <https://github.com/zalandoresearch/fashion-mnist>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``60 000`` for train, ``10 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    train_eval_size (int): Size of the train eval data set.
        Defaults to ``10 000`` the size of the test set.
  """

    def __init__(self,
                 batch_size,
                 train_eval_size=10000):
        """Creates a new Fashion-MNIST instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``60 000`` for train, ``10 000``
          for test) the remainder is dropped in each epoch (after shuffling).
      train_eval_size (int): Size of the train eval data set.
          Defaults to ``10 000`` the size of the test set.
    """
        self._name = "fmnist"
        self._train_eval_size = train_eval_size
        super(fmnist, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        transform = transforms.ToTensor()
        train_dataset = datasets.FashionMNIST(root=config.get_data_dir(), train=True, download=True, transform=transform)
        valid_dataset = datasets.FashionMNIST(root=config.get_data_dir(), train=True, download=True, transform=transform)
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(train_dataset, valid_dataset)
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = transforms.ToTensor()
        test_dataset = datasets.FashionMNIST(root=config.get_data_dir(), train=False, download=True, transform=transform)
        return self._make_dataloader(test_dataset, sampler=None)
