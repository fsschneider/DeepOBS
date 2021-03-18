"""SVHN DeepOBS data set for PyTorch."""

import os

from torchvision import datasets, transforms

from deepobs.config import get_data_dir
from deepobs.datasets.pytorch._dataset import DataSet


class SVHN(DataSet):
    """DeepOBS data set class for the `Street View House Numbers (SVHN)\
    <http://ufldl.stanford.edu/housenumbers/>`_ data set.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in each
            epoch (after shuffling).
        data_augmentation (bool): If ``True`` some data augmentation operations
            (random crop window, lighting augmentation) are applied to the
            training data (but not the test data).
        train_eval_size (int): Size of the train eval data set.
            Defaults to ``26,032`` the size of the test set.
    """

    def __init__(self, batch_size, data_augmentation=True, train_eval_size=26032):
        """Creates a new SVHN instance.

        Args:
            batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                is not a divider of the dataset size the remainder is dropped in each
                epoch (after shuffling).
            data_augmentation (bool): If ``True`` some data augmentation operations
                (random crop window, lighting augmentation) are applied to the
                training data (but not the test data).
            train_eval_size (int): Size of the train eval data set.
                Defaults to ``26,032`` the size of the test set.
        """
        self._data_augmentation = data_augmentation
        self._transform_no_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4376821, 0.4437697, 0.47280442),
                    (0.19803012, 0.20101562, 0.19703614),
                ),
            ]
        )

        self._transform_data_augmentation = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.RandomCrop(size=(32, 32)),
                transforms.ColorJitter(
                    brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4376821, 0.4437697, 0.47280442),
                    (0.19803012, 0.20101562, 0.19703614),
                ),
            ]
        )
        super().__init__(batch_size, train_eval_size)

    def _make_train_and_valid_dataloader(self):
        if self._data_augmentation:
            transform = self._transform_data_augmentation
        else:
            transform = self._transform_no_augmentation

        train_dataset = datasets.SVHN(
            root=os.path.join(get_data_dir(), self._name),
            split="train",
            download=True,
            transform=transform,
        )
        # we want the validation set to be of the same size as the test set,
        # so we do NOT use the 'extra' dataset that is available for SVHN
        valid_dataset = datasets.SVHN(
            root=os.path.join(get_data_dir(), self._name),
            split="train",
            download=True,
            transform=self._transform_no_augmentation,
        )
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
            train_dataset, valid_dataset, shuffle_dataset=True
        )
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = self._transform_no_augmentation
        test_dataset = datasets.SVHN(
            root=os.path.join(get_data_dir(), self._name),
            split="test",
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)
