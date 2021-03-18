"""MNIST DeepOBS data set for PyTorch."""

from torchvision import datasets, transforms

from deepobs.config import get_data_dir
from deepobs.datasets.pytorch._dataset import DataSet


class MNIST(DataSet):
    """DeepOBS data set class for the `MNIST\
    <http://yann.lecun.com/exdb/mnist/>`_ data set.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in each
            epoch (after shuffling).
        train_eval_size (int): Size of the train eval data set.
            Defaults to ``10,000`` the size of the test set.
    """

    def __init__(self, batch_size, train_eval_size=10000):
        """Creates a new MNIST instance.

        Args:
            batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                is not a divider of the dataset size the remainder is dropped in each
                epoch (after shuffling).
            train_eval_size (int): Size of the train eval data set.
                Defaults to ``10,000`` the size of the test set.
        """
        super().__init__(batch_size, train_eval_size)

    def _make_train_and_valid_dataloader(self):
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            root=get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        valid_dataset = datasets.MNIST(
            root=get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
            train_dataset, valid_dataset, shuffle_dataset=True
        )
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = transforms.ToTensor()
        test_dataset = datasets.MNIST(
            root=get_data_dir(),
            train=False,
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)
