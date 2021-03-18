"""Quadratic DeepOBS data set for PyTorch."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from deepobs.datasets.pytorch._dataset import DataSet


class Quadratic(DataSet):
    """DeepOBS data set class for a toy data set for a stochastic quadratic problem.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in each
            epoch (after shuffling).
        dim (int, optional): Dimensionality of the quadratic. Defaults to ``100``.
        train_size (int): Size of the dataset; will be used for train, train eval
            and test datasets. Defaults to ``1000``.
        noise_level (float): Standard deviation of the data points around the mean.
            The data points are drawn from a Gaussian distribution.
            Defaults to ``0.6``.
    """

    def __init__(self, batch_size, dim=100, train_size=1000, noise_level=0.6):
        """Creates a new Quadratic instance.

        Args:
            batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                is not a divider of the dataset size the remainder is dropped in each
                epoch (after shuffling).
            dim (int, optional): Dimensionality of the quadratic. Defaults to ``100``.
            train_size (int): Size of the dataset; will be used for train, train eval
                and test datasets. Defaults to ``1000``.
            noise_level (float): Standard deviation of the data points around the mean.
                The data points are drawn from a Gaussian distribution.
                Defaults to ``0.6``.
        """
        self._dim = dim
        self._train_size = train_size
        self._noise_level = noise_level
        super().__init__(batch_size, train_size)

    def _make_labels(self):
        """Return zeros as labels."""
        np_labels = np.zeros((self._train_size, self._dim), dtype=np.float32)
        return torch.from_numpy(np_labels)

    def _make_data(self, seed):
        """Draw the data.

        We draw the data from a random generator with a fixed seed to always get
        the same data and add noise.

        Args:
            seed (int): Random seed to use for drawing the data.

        Returns:
            Tensor: Tensor holding the data.
        """
        rng = np.random.RandomState(seed)
        X = rng.normal(0.0, self._noise_level, (self._train_size, self._dim))
        X = np.float32(X)
        return torch.from_numpy(X)

    def _make_train_and_valid_dataloader(self):
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        X = self._make_data(seed=42)
        Y = self._make_labels()
        train_dataset = TensorDataset(X, Y)

        X = self._make_data(seed=44)
        Y = self._make_labels()
        valid_dataset = TensorDataset(X, Y)

        train_loader = self._make_dataloader(train_dataset, shuffle=True)
        valid_loader = self._make_dataloader(valid_dataset)
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        X = self._make_data(seed=43)
        Y = self._make_labels()
        test_dataset = TensorDataset(X, Y)
        return self._make_dataloader(test_dataset)

    def _make_train_eval_dataloader(self):
        return self._train_dataloader
