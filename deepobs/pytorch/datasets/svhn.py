# -*- coding: utf-8 -*-
"""SVHN DeepOBS dataset."""

from . import dataset
from .. import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms
from .datasets_utils import train_eval_sampler


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

    def _make_dataloader(self, split, shuffle=True, data_augmentation = False, sampler=None):
        if data_augmentation:
            transform = transforms.Compose([
                    transforms.Pad(padding=2),
                    transforms.RandomCrop(size=(32,32)),
                    transforms.ColorJitter(brightness=63. / 255., saturation=[0.5,1.5], contrast=[0.2,1.8]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4376821,  0.4437697,  0.47280442), (0.19803012, 0.20101562, 0.19703614))
                    ])
        else:
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4376821,  0.4437697,  0.47280442), (0.19803012, 0.20101562, 0.19703614))
                    ])

        dataset = datasets.SVHN(root = config.get_data_dir(), split = split, download = True, transform = transform)
        loader = dat.DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, drop_last=True, pin_memory=self._pin_memory, num_workers=self._num_workers, sampler=sampler)
        return loader

    def _make_train_dataloader(self):
        return self._make_dataloader(split='train', shuffle = True, data_augmentation = self._data_augmentation, sampler=None)

    def _make_test_dataloader(self):
        return self._make_dataloader(split='test', shuffle = False, data_augmentation = False, sampler=None)

    def _make_train_eval_dataloader(self):
        size = len(self._train_dataloader.dataset)
        sampler = train_eval_sampler(size, self._train_eval_size)
        return self._make_dataloader(split='train', shuffle=False, data_augmentation=self._data_augmentation, sampler=sampler)