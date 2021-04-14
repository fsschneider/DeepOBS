"""Base class for DeepOBS data sets in PyTorch."""

from abc import ABC, abstractmethod

import numpy as np
from torch import randperm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

from deepobs.config import get_default_device, get_num_workers


class DataSet(ABC):
    """Base class for DeepOBS data sets in PyTorch.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size (``60 000`` for train, ``10 000``
            for test) the remainder is dropped in each epoch (after shuffling).
        train_eval_size (int): Size of the train eval data set.

    Methods:
         _make_train_and_valid_dataloader: Creates a PyTorch DataLoader for the
            training and validation data with batches of size batch_size.
         _make_train_eval_dataloader: Creates a PyTorch DataLoader for the
            training evaluation data with batches of size batch_size.
         _make_test_dataloader: Creates a PyTorch DataLoader for the
            test data with batches of size batch_size.

    Attributes:
        _batch_size: The mini-batch size to use.
        _train_eval_size (int): Size of the train eval data set.
        _name: Name of the data set.
        _pin_memory: Whether to pin memory for the dataloaders.
            Defaults to 'False' if 'cuda' is not the current device.
        _num_workers: The number of workers used for the dataloaders.
            It's value is set to the global variable NUM_WORKERS.
        _train_dataloader: A PyTorch DataLoader instance holding the training data.
        _valid_dataloader: A PyTorch DataLoader instance holding the validation data.
        _train_eval_dataloader: A PyTorch DataLoader instance holding the training
            data for evaluation. This is the same as the ``_train_dataloader``,
            but using the same number of samples as the test set.
        _test_dataloader: A PyTorch DataLoader instance holding the test data.
    """

    def __init__(self, batch_size, train_eval_size):
        """Creates the new DataSet instance, by building the DataLoaders.

        Args:
            batch_size (int): The mini-batch size to use.
            train_eval_size (int): Size of the train eval data set.
        """
        self._batch_size = batch_size
        self._train_eval_size = train_eval_size
        self._name = self.__class__.__name__

        # Set pin_memory if CUDA is used, as discussed in
        # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        if "cuda" in get_default_device():
            self._pin_memory = True
        else:
            self._pin_memory = False

        self._num_workers = get_num_workers()

        # Build the four dataloader (train, train_eval, valid, test)
        (
            self._train_dataloader,
            self._valid_dataloader,
        ) = self._make_train_and_valid_dataloader()
        self._train_eval_dataloader = self._make_train_eval_dataloader()
        self._test_dataloader = self._make_test_dataloader()

    @abstractmethod
    def _make_train_and_valid_dataloader(self):
        """Creates the training and validation DataLoader.

        Raises:
            NotImplementedError: If not implemented. Should be defined by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _make_test_dataloader(self):
        """Creates the test DataLoader.

        Raises:
            NotImplementedError: If not implemented. Should be defined by subclass.
        """
        raise NotImplementedError

    def _make_train_eval_dataloader(self):
        """Helper function to create the training evaluation data loader.

        Returns:
            DataLoader: A DataLoader instance with batches of training eval data.
        """
        size = len(self._train_dataloader.dataset)
        sampler = _train_eval_sampler(size, self._train_eval_size)
        return self._make_dataloader(self._train_dataloader.dataset, sampler=sampler)

    def _make_dataloader(self, dataset, sampler=None, shuffle=False):
        """Helper function to build a DataLoader from a data set.

        Args:
            dataset (Dataset): dataset from which to load the data.
            sampler (Sampler or None, optional): Defines the strategy to draw samples
            shuffle (bool, optional): set to ``True`` to reshuffle at every epoch.
                Defaults to False.

        Returns:
            DataLoader: A torch.utils.data.DataLoader instance.
        """
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            drop_last=True,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            sampler=sampler,
            shuffle=shuffle,
        )
        return loader

    def _make_train_valid_split_sampler(self, train_dataset, shuffle_dataset):
        """Helper function to build samplers for the train and validation set.

        Args:
            train_dataset (Dataset): Full data set that should be split into the
                training and validation dataset.
            shuffle_dataset (bool): Whether the data set should be shuffled
                before splitting.

        Returns:
            Sampler: A random samplers for the train set.
            Sampler: A random samplers for the validation set.
        """
        indices = list(range(len(train_dataset)))
        if shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, valid_indices = (
            indices[self._train_eval_size :],
            indices[: self._train_eval_size],
        )
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        return train_sampler, valid_sampler

    def _make_train_and_valid_dataloader_helper(
        self, train_dataset, valid_dataset, shuffle_dataset
    ):
        """Helper function to split the (full) train data into a train and valid set.

        Args:
            train_dataset (Dataset): Data to use for the train set. Usually this
                is the full training set and parts of it will be used as the actual
                training data.
            valid_dataset (Dataset):  Data to use for the validation set. Usually this
                is (again) the full training set and everything that was NOT used
                for the train data will be used here.
            shuffle_dataset (bool): Whether the data set should be shuffled
                before splitting.

        Returns:
            DataLoader: A DataLoader for the train set.
            DataLoader: A DataLoader for the validation set.
        """
        train_sampler, valid_sampler = self._make_train_valid_split_sampler(
            train_dataset, shuffle_dataset
        )
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(train_dataset, sampler=train_sampler)
        valid_loader = self._make_dataloader(valid_dataset, sampler=valid_sampler)
        return train_loader, valid_loader


class _train_eval_sampler(Sampler):
    """A helper subclass of torch Sampler to easily draw the train eval set."""

    def __init__(self, size, sub_size):
        """Initialize the train eval sampler.

        Args:
            size (int): The size of the original dataset.
            sub_size (int): The size of the dataset after sampling.
        """
        self.size = size
        self.sub_size = sub_size

    def __iter__(self):
        indices = randperm(self.size).tolist()
        sub_indices = indices[0 : self.sub_size]
        return iter(sub_indices)

    def __len__(self):
        return self.sub_size
