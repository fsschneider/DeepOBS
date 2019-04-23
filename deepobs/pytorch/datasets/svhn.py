# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""SVHN DeepOBS dataset."""

import numpy as np
from . import dataset
from .. import config
from torch.utils import data as dat
from torchvision import datasets
from torchvision import transforms


class svhn(dataset.DataSet):


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
        loader = dat.DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, drop_last=True, pin_memory=True, num_workers=4, sampler=sampler)
        return loader

    def _make_train_dataloader(self):
        return self._make_dataloader(split='train', shuffle = True, data_augmentation = self._data_augmentation, sampler=None)

    def _make_test_dataloader(self):
        return self._make_dataloader(split='test', shuffle = False, data_augmentation = False, sampler=None)

    def _make_train_eval_dataloader(self):
        indices = np.random.choice(len(self._train_dataloader.dataset), size= self._train_eval_size, replace=False)
        sampler = dat.SubsetRandomSampler(indices)
        return self._make_dataloader(split='train', shuffle=False, data_augmentation=False, sampler=sampler)