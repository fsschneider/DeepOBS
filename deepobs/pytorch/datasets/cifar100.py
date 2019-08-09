# -*- coding: utf-8 -*-
"""CIFAR-100 DeepOBS dataset."""
from . import dataset
from deepobs import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms
from .datasets_utils import train_eval_sampler

training_transform_augmented = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.RandomCrop(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
        transforms.ToTensor(),
        transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
    ])

training_transform_not_augmented = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
])


class cifar100(dataset.DataSet):
    """DeepOBS data set class for the `CIFAR-100\
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``50 000`` for train, ``10 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    data_augmentation (bool): If ``True`` some data augmentation operations
        (random crop window, horizontal flipping, lighting augmentation) are
        applied to the training data (but not the test data).
    train_eval_size (int): Size of the train eval data set.
        Defaults to ``10 000`` the size of the test set.

  Methods:
      _make_dataloader: A helper that is shared by all three data loader methods.
  """

    def __init__(self,
                 batch_size,
                 data_augmentation=True,
                 train_eval_size=10000):
        """Creates a new CIFAR-100 instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``50 000`` for train, ``10 000``
          for test) the remainder is dropped in each epoch (after shuffling).
      data_augmentation (bool): If ``True`` some data augmentation operations
          (random crop window, horizontal flipping, lighting augmentation) are
          applied to the training data (but not the test data).
      train_eval_size (int): Size of the train eval data set.
          Defaults to ``10 000`` the size of the test set.
    """
        self._name = "cifar100"
        self._data_augmentation = data_augmentation
        self._train_eval_size = train_eval_size
        super(cifar100, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        if self._data_augmentation:
            transform = training_transform_augmented
        else:
            transform = training_transform_not_augmented

        train_dataset = datasets.CIFAR100(root=config.get_data_dir(), train=True, download=True, transform=transform)
        valid_dataset = datasets.CIFAR100(root=config.get_data_dir(), train=True, download=True, transform=training_transform_not_augmented)
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(train_dataset, valid_dataset)
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = training_transform_not_augmented
        test_dataset = datasets.CIFAR100(root=config.get_data_dir(), train=False, download=True, transform=transform)
        return self._make_dataloader(test_dataset, sampler=None)
