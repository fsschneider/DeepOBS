"""Two-D DeepOBS data set for PyTorch."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from deepobs.datasets.pytorch._dataset import DataSet


class TwoD(DataSet):
    """DeepOBS data set class for two dimensional stochastic toy problems.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in each
            epoch (after shuffling).
        train_size (int): Size of the dataset; will be used for train, train eval
            and test datasets. Defaults to ``10000``.
        noise_level (float): Standard deviation of the data points around the mean.
            The data points are drawn from a Gaussian distribution.
            Defaults to ``1.0``.
    """

    def __init__(self, batch_size, train_size=10000, noise_level=1.0):
        """Creates a new Quadratic instance.

        Args:
            batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                is not a divider of the dataset size the remainder is dropped in each
                epoch (after shuffling).
            train_size (int): Size of the dataset; will be used for train, train eval
                and test datasets. Defaults to ``10000``.
            noise_level (float): Standard deviation of the data points around the mean.
                The data points are drawn from a Gaussian distribution.
                Defaults to ``1.0``.
        """
        self._train_size = train_size
        self._noise_level = noise_level
        super().__init__(batch_size, train_size)

    def _make_data(self, seed):
        """Draw the data.

        We draw the data from a random generator with a fixed seed to always get
        the same data and add noise.

        Args:
            seed (int): Random seed to use for drawing the data.

        Returns:
            Tensor: Tensor holding the data.
            Tensor: Tensor holding the "labels".
        """
        rng = np.random.RandomState(seed)
        X = rng.normal(0.0, self._noise_level, self._train_size)
        Y = rng.normal(0.0, self._noise_level, self._train_size)
        X = np.float32(X)
        Y = np.float32(Y)
        return torch.from_numpy(X), torch.from_numpy(Y)

    def _make_train_and_valid_dataloader(self):
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        X, Y = self._make_data(seed=42)
        train_dataset = TensorDataset(X, Y)

        X, Y = self._make_data(seed=44)
        valid_dataset = TensorDataset(X, Y)

        train_loader = self._make_dataloader(train_dataset, shuffle=True)
        valid_loader = self._make_dataloader(valid_dataset)
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        X, Y = self._make_data(seed=43)
        test_dataset = TensorDataset(X, Y)
        return self._make_dataloader(test_dataset)

    def _make_train_eval_dataloader(self):
        return self._train_dataloader
