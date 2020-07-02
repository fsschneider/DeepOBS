# -*- coding: utf-8 -*-
"""MNIST DeepOBS dataset."""

from __future__ import print_function

from torch.utils import data as dat
from torchvision import datasets, transforms

from deepobs import config

from . import dataset
from .datasets_utils import train_eval_sampler

transform_images_resize = transforms.Compose(
    [transforms.Resize(64),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),
     ]
)

transform_images_no_resize = transforms.Compose(
    [transforms.ToTensor()
     ]
)


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

    def __init__(self, batch_size, resize_images=False, train_eval_size=10000):
        """Creates a new MNIST instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``60 000`` for train, ``10 000``
          for test) the remainder is dropped in each epoch (after shuffling).
      train_eval_size (int): Size of the train eval data set.
          Defaults to ``10 000`` the size of the test set.
    """
        self._name = "mnist"
        self._resize_images = resize_images
        self._train_eval_size = train_eval_size
        super(mnist, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        if self._resize_images:
            transform = transform_images_resize
        else:
            transform = transform_images_no_resize
        train_dataset = datasets.MNIST(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        valid_dataset = datasets.MNIST(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
            train_dataset, valid_dataset
        )
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        if self._resize_images:
            transform = transform_images_resize
        else:
            transform = transform_images_no_resize
        test_dataset = datasets.MNIST(
            root=config.get_data_dir(),
            train=False,
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)
