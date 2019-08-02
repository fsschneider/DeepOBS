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

    def _make_train_and_valid_dataloader(self):
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        rng = np.random.RandomState(42)
        X = rng.normal(0.0, self._noise_level, (self._train_size, self._dim))
        X = np.float32(X)
        train_dataset = dat.TensorDataset(torch.from_numpy(X))

        rng = np.random.RandomState(44)
        X = rng.normal(0.0, self._noise_level, (self._train_size, self._dim))
        X = np.float32(X)
        valid_dataset = dat.TensorDataset(torch.from_numpy(X))

        train_loader = self._make_dataloader(train_dataset, shuffle=True)
        valid_loader = self._make_dataloader(valid_dataset)
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        rng = np.random.RandomState(43)
        X = rng.normal(0.0, self._noise_level, (self._train_size, self._dim))
        X = np.float32(X)
        test_dataset = dat.TensorDataset(torch.from_numpy(X))
        return self._make_dataloader(test_dataset)

    def _make_train_eval_dataloader(self):
        return self._train_dataloader
