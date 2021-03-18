"""CIFAR DeepOBS data sets for PyTorch."""

import os

from torchvision import datasets, transforms

from deepobs.config import get_data_dir
from deepobs.datasets.pytorch._dataset import DataSet


class CIFAR10(DataSet):
    """DeepOBS data set class for the `CIFAR-10\
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_ data set.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in each
            epoch (after shuffling).
        data_augmentation (bool): If ``True`` some data augmentation operations
            (random crop window, horizontal flipping, lighting augmentation) are
            applied to the training data (but not the test data).
        train_eval_size (int): Size of the train eval data set.
                Defaults to ``10,000`` the size of the test set.
    """

    def __init__(self, batch_size, data_augmentation=True, train_eval_size=10000):
        """Creates a new CIFAF10 instance.

        Args:
            batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                is not a divider of the dataset size the remainder is dropped in each
                epoch (after shuffling).
            data_augmentation (bool): If ``True`` some data augmentation operations
                (random crop window, horizontal flipping, lighting augmentation) are
                applied to the training data (but not the test data).
            train_eval_size (int): Size of the train eval data set.
                Defaults to ``10,000`` the size of the test set.
        """
        self._data_augmentation = data_augmentation
        self._transform_no_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784),
                ),
            ]
        )

        self._transform_data_augmentation = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.RandomCrop(size=(32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784),
                ),
            ]
        )
        super().__init__(batch_size, train_eval_size)

    def _make_train_and_valid_dataloader(self):
        if self._data_augmentation:
            transform = self._transform_data_augmentation
        else:
            transform = self._transform_no_augmentation

        train_dataset = datasets.CIFAR10(
            root=os.path.join(get_data_dir(), self._name),
            train=True,
            download=True,
            transform=transform,
        )
        valid_dataset = datasets.CIFAR10(
            root=os.path.join(get_data_dir(), self._name),
            train=True,
            download=True,
            transform=self._transform_no_augmentation,
        )
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
            train_dataset, valid_dataset, shuffle_dataset=True
        )
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = self._transform_no_augmentation
        test_dataset = datasets.CIFAR10(
            root=os.path.join(get_data_dir(), self._name),
            train=False,
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)


class CIFAR100(DataSet):
    """DeepOBS data set class for the `CIFAR-100\
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_ data set.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in each
            epoch (after shuffling).
        data_augmentation (bool): If ``True`` some data augmentation operations
            (random crop window, horizontal flipping, lighting augmentation) are
            applied to the training data (but not the test data).
        train_eval_size (int): Size of the train eval data set.
            Defaults to ``10,000`` the size of the test set.
    """

    def __init__(self, batch_size, data_augmentation=True, train_eval_size=10000):
        """Creates a new CIFAF10 instance.

        Args:
            batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                is not a divider of the dataset size the remainder is dropped in each
                epoch (after shuffling).
            data_augmentation (bool): If ``True`` some data augmentation operations
                (random crop window, horizontal flipping, lighting augmentation) are
                applied to the training data (but not the test data).
            train_eval_size (int): Size of the train eval data set.
                Defaults to ``10,000`` the size of the test set.
        """
        self._data_augmentation = data_augmentation
        self._transform_no_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.50707516, 0.48654887, 0.44091784),
                    (0.26733429, 0.25643846, 0.27615047),
                ),
            ]
        )

        self._transform_data_augmentation = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.RandomCrop(size=(32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.50707516, 0.48654887, 0.44091784),
                    (0.26733429, 0.25643846, 0.27615047),
                ),
            ]
        )
        super().__init__(batch_size, train_eval_size)

    def _make_train_and_valid_dataloader(self):
        if self._data_augmentation:
            transform = self._transform_data_augmentation
        else:
            transform = self._transform_no_augmentation

        train_dataset = datasets.CIFAR100(
            root=os.path.join(get_data_dir(), self._name),
            train=True,
            download=True,
            transform=transform,
        )
        valid_dataset = datasets.CIFAR100(
            root=os.path.join(get_data_dir(), self._name),
            train=True,
            download=True,
            transform=self._transform_no_augmentation,
        )
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
            train_dataset, valid_dataset, shuffle_dataset=True
        )
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = self._transform_no_augmentation
        test_dataset = datasets.CIFAR100(
            root=os.path.join(get_data_dir(), self._name),
            train=False,
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)
