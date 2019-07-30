# -*- coding: utf-8 -*-
"""CIFAR-10 DeepOBS dataset."""

from . import dataset
from .. import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from .datasets_utils import train_eval_sampler

training_transform_not_augmented = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513, 0.26158784))
                    ])

training_transform_augmented = transforms.Compose([
                    transforms.Pad(padding=2),
                    transforms.RandomCrop(size=(32,32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=63. / 255., saturation=[0.5,1.5], contrast=[0.2,1.8]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513, 0.26158784))
                    ])

class cifar10(dataset.DataSet):
    """DeepOBS data set class for the `CIFAR-10\
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
        """Creates a new CIFAR-10 instance.

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
        self._name = "cifar10"
        self._data_augmentation = data_augmentation
        self._train_eval_size = train_eval_size
        super(cifar10, self).__init__(batch_size)

    def _make_dataloader(self, dataset, sampler = None):
        loader = dat.DataLoader(dataset, batch_size=self._batch_size, drop_last=True,
                                pin_memory=self._pin_memory, num_workers=self._num_workers, sampler=sampler)
        return loader

    def _make_train_and_valid_dataloader(self):
        if self._data_augmentation:
            transform = training_transform_augmented
        else:
            transform = training_transform_not_augmented

        train_dataset = datasets.CIFAR10(root=config.get_data_dir(), train=True, download=True, transform=transform)
        valid_dataset = datasets.CIFAR10(root=config.get_data_dir(), train=True, download=True, transform=training_transform_not_augmented)
        indices = list(range(len(train_dataset)))
        train_indices, valid_indices = indices[self._train_eval_size:], indices[:self._train_eval_size]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(train_dataset, sampler=train_sampler)
        valid_loader = self._make_dataloader(valid_dataset, sampler=valid_sampler)
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        # TODO what are the transforms for the test set? what is the normalization? the one of train set? or all?
        transform = training_transform_not_augmented
        test_dataset = datasets.CIFAR10(root=config.get_data_dir(), train=False, download=True, transform=transform)
        return self._make_dataloader(test_dataset, sampler=None)

    def _make_train_eval_dataloader(self):
        size = len(self._train_dataloader.dataset)
        sampler = train_eval_sampler(size, self._train_eval_size)
        # TODO is it correct that train eval set is not augmented in the same way like the train set?
        return self._make_dataloader(self._train_dataloader.dataset, sampler=sampler)