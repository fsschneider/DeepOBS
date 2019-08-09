# -*- coding: utf-8 -*-
"""SVHN DeepOBS dataset."""

from . import dataset
from deepobs import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms
from .datasets_utils import train_eval_sampler

training_transform_augmented = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.RandomCrop(size=(32, 32)),
        transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])

training_transform_not_augmented = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])


class svhn(dataset.DataSet):
    """DeepOBS data set class for the `Street View House Numbers (SVHN)\
    <http://ufldl.stanford.edu/housenumbers/>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``73 000`` for train, ``26 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    data_augmentation (bool): If ``True`` some data augmentation operations
        (random crop window, lighting augmentation) are applied to the
        training data (but not the test data).
    train_eval_size (int): Size of the train eval dataset.
        Defaults to ``26 000`` the size of the test set.
  """
    def __init__(self,
                 batch_size,
                 data_augmentation=True,
                 train_eval_size=26032):

        self._name = "svhn"
        self._data_augmentation = data_augmentation
        self._train_eval_size = train_eval_size
        super(svhn, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        if self._data_augmentation:
            transform = training_transform_augmented
        else:
            transform = training_transform_not_augmented
        train_dataset = datasets.SVHN(root=config.get_data_dir(), split='train', download=True, transform=transform)
        # we want the validation set to be of the same size as the test set, so we do NOT use the 'extra' dataset that is available for SVHN
        valid_dataset = datasets.SVHN(root=config.get_data_dir(), split='train', download=True, transform=training_transform_not_augmented)
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(train_dataset, valid_dataset)
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = training_transform_not_augmented
        test_dataset = datasets.SVHN(root=config.get_data_dir(), split='test', download=True, transform=transform)
        return self._make_dataloader(test_dataset, sampler=None)