# -*- coding: utf-8 -*-
"""Quadratic DeepOBS dataset."""

import numpy as np
from . import dataset
from torch.utils import data as dat
import torch

class quadratic(dataset.DataSet):
    """DeepOBS data set class to create an n dimensional stochastic quadratic\
    testproblem.

    This toy data set consists of a fixed number (``train_size``) of iid draws
    from a zero-mean normal distribution in ``dim`` dimensions with isotropic
    covariance specified by ``noise_level``.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``1000`` for train and test) the
        remainder is dropped in each epoch (after shuffling).
    dim (int): Dimensionality of the quadratic. Defaults to ``100``.
    train_size (int): Size of the dataset; will be used for train, train eval and
        test datasets. Defaults to ``1000``.
    noise_level (float): Standard deviation of the data points around the mean.
        The data points are drawn from a Gaussian distribution.
        Defaults to ``0.6``.

  Methods:
      _make_dataloader: A helper that is shared by all three data loader methods.
  """

    def __init__(self, batch_size, dim=100, train_size=1000, noise_level=0.6):
        """Creates a new Quadratic instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``1000`` for train and test) the
          remainder is dropped in each epoch (after shuffling).
      dim (int): Dimensionality of the quadratic. Defaults to ``100``.
      train_size (int): Size of the dataset; will be used for train, train eval
          and test datasets. Defaults to ``1000``.
      noise_level (float): Standard deviation of the data points around the mean.
          The data points are drawn from a Gaussian distribution.
          Defaults to ``0.6``.
    """
        self._name = "quadratic"
        self._dim = dim
        self._train_size = train_size
        self._noise_level = noise_level
        super(quadratic, self).__init__(batch_size)

    def _make_dataloader(self, X, shuffle=True):
        """Creates a quadratic data set (helper used by ``.make_*_datset`` below).

        Args:
            X (np.array): Numpy array containing the ``x`` values of the data points.
            data_y (np.array): Numpy array containing the ``y`` values of the data points.
            shuffle (bool):  Switch to turn on or off shuffling of the data set.
                Defaults to ``True``.

        Returns:
            A tf.data.Dataset yielding batches of quadratic data.
        """


        dataset = dat.TensorDataset(torch.from_numpy(X))
        loader = dat.DataLoader(dataset=dataset, batch_size=self._batch_size, shuffle=shuffle, drop_last=True, pin_memory = self._pin_memory, num_workers = self._num_workers)
        return loader

    def _make_train_dataloader(self):
        """Creates the quadratic training dataset.

    Returns:
      A torch.utils.data.DataLoader instance with batches of training data.
    """
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        rng = np.random.RandomState(42)
        X = rng.normal(0.0, self._noise_level, (self._train_size, self._dim))
        X = np.float32(X)
        return self._make_dataloader(X, shuffle=True)

    def _make_train_eval_dataloader(self):
        """Creates the quadratic train eval dataset.

        Returns:
            A torch.utils.data.DataLoader instance with batches of training eval data.
        """
        # take whole train set for train evaluation
        return self._train_dataloader

    def _make_test_dataloader(self):
        """Creates the quadratic test dataset.

        Returns:
            A torch.utils.data.DataLoader instance with batches of test data.
        """
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        rng = np.random.RandomState(43)
        X = rng.normal(0.0, self._noise_level, (self._train_size, self._dim))
        X = np.float32(X)
        return self._make_dataloader(X, shuffle=False)
