# -*- coding: utf-8 -*-
import abc
from .. import config
from torch.utils import data as dat
from torch.utils.data.sampler import SubsetRandomSampler
from .datasets_utils import train_eval_sampler
from random import shuffle

"""Base class for DeepOBS datasets."""

# pylint: disable=too-many-instance-attributes, too-few-public-methods


class DataSet(abc.ABC):
    """Base class for DeepOBS data sets.

    Args:
        batch_size (int): The mini-batch size to use.

    Methods:
         _make_train_and_valid_dataloader: Creates a torch data loader for the training and validation data with batches of size batch_size.
         _make_train_eval_dataloader: Creates a torch data loader for the training evaluation data with batches of size batch_size.
         _make_test_dataloader: Creates a torch data loader for the test data with batches of size batch_size.

    Attributes:
        _pin_memory: Whether to pin memory for the dataloaders. Defaults to 'False' if 'cuda' is not the current device.
        _num_workers: The number of workers used for the dataloaders. It's value is set to the global variable NUM_WORKERS.
        _train_dataloader: A torch.utils.data.DataLoader instance that holds the training data.
        _valid_dataloader: A torch.utils.data.DataLoader instance that holds the validation data.
        _train_eval_dataloader: A torch.utils.data.DataLoader instance that holds the training evaluation data.
        _test_dataloader: A torch.utils.data.DataLoader instance that holds the test data.
  """

    def __init__(self, batch_size):
        """Creates a new DataSet instance.

    Args:
      batch_size (int): The mini-batch size to use.
    """
        self._batch_size = batch_size

        if "cuda" in config.get_default_device():
            self._pin_memory = True
        else:
            self._pin_memory = False
        self._num_workers = config.get_num_workers()
        self._train_dataloader, self._valid_dataloader = (
            self._make_train_and_valid_dataloader()
        )
        self._train_eval_dataloader = self._make_train_eval_dataloader()
        self._test_dataloader = self._make_test_dataloader()

    def _make_dataloader(self, dataset, sampler=None, shuffle=False):
        loader = dat.DataLoader(
            dataset,
            batch_size=self._batch_size,
            drop_last=True,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            sampler=sampler,
            shuffle=shuffle,
        )
        return loader

    def _make_train_eval_split_sampler(self, train_dataset):
        """Generates SubSetRandomSamplers that can be used for splitting the training set."""
        indices = list(range(len(train_dataset)))
        shuffle(indices)
        train_indices, valid_indices = (
            indices[self._train_eval_size :],
            indices[: self._train_eval_size],
        )
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        return train_sampler, valid_sampler

    def _make_train_and_valid_dataloader_helper(
        self, train_dataset, valid_dataset
    ):
        train_sampler, valid_sampler = self._make_train_eval_split_sampler(
            train_dataset
        )
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(
            train_dataset, sampler=train_sampler
        )
        valid_loader = self._make_dataloader(
            valid_dataset, sampler=valid_sampler
        )
        return train_loader, valid_loader

    def _make_train_eval_dataloader(self):
        """Creates the training evaluation data loader.

        Returns:
          A torch.utils.data.DataLoader instance with batches of training evaluatoion data.
        """
        size = len(self._train_dataloader.dataset)
        sampler = train_eval_sampler(size, self._train_eval_size)
        return self._make_dataloader(
            self._train_dataloader.dataset, sampler=sampler
        )

    @abc.abstractmethod
    def _make_train_and_valid_dataloader(self):
        """Creates the training and validation data loader.

    Returns:
      A torch.utils.data.DataLoader instance with batches of training data.
      A torch.utils.data.DataLoader instance with batches of validation data.
    """
        pass

    @abc.abstractmethod
    def _make_test_dataloader(self):
        """Creates the test data loader.

    Returns:
      A torch.utils.data.DataLoader instance with batches of test data.
    """
        pass
