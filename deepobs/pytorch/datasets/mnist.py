# -*- coding: utf-8 -*-
"""MNIST DeepOBS dataset."""

from __future__ import print_function

import numpy as np
from . import dataset
from .. import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms

class mnist(dataset.DataSet):
    """DeepOBS data set class for the `MNIST\
    <http://yann.lecun.com/exdb/mnist/>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``60 000`` for train, ``10 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    train_eval_size (int): Size of the train eval data set.
        Defaults to ``10 000`` the size of the test set.

  Attributes:
    batch: A tuple ``(x, y)`` of tensors, yielding batches of MNIST images
        (``x`` with shape ``(batch_size, 28, 28, 1)``) and corresponding one-hot
        label vectors (``y`` with shape ``(batch_size, 10)``). Executing these
        tensors raises a ``tf.errors.OutOfRangeError`` after one epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to ``train``, ``train_eval``
        or ``test``, depending on the current phase. This can be used by testproblems
        to adapt their behavior to this phase.
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

    def _make_dataloader(self, train):
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root = config.get_data_dir(), train = train, download = True, transform = transform)
        loader = dat.DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        return loader

    def _make_train_dataloader(self):
        return self._make_dataloader(train=True)

    def _make_test_dataloader(self):
        return self._make_dataloader(train=False)

    def _make_train_eval_dataloader(self):
        indices = np.random.choice(len(self._train_dataloader.dataset), size= self._train_eval_size, replace=False)
        sampler = dat.SubsetRandomSampler(indices)
        loader = dat.DataLoader(self._train_dataloader.dataset, batch_size=self._batch_size, drop_last=True, sampler=sampler)
        return loader