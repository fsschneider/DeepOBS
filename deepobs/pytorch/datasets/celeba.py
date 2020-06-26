# -*- coding: utf-8 -*-
"""CelebA DeepOBS dataset."""

import os
from torchvision.transforms import transforms
from torchvision import datasets, transforms
# from torchvision.datasets.celeb import CelebA

from deepobs import config

from . import dataset


transform_images_resize = transforms.Compose(
                [transforms.Resize(64),
                 transforms.CenterCrop(64),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 ]
            )

transform_images_no_resize = transforms.Compose(
    [transforms.ToTensor()
     ]
)


class celeba(dataset.DataSet):
    """DeepOBS data set class for the `\
    <Put link here>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``73 000`` for train, ``26 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    resize_images (bool): If ``True`` some data augmentation operations
        (random crop window, lighting augmentation) are applied to the
        training data (but not the test data).
    train_eval_size (int): Size of the train eval dataset.
        Defaults to ``26 000`` the size of the test set.
  """

    def __init__(self, batch_size, resize_images=False, train_eval_size=10000):
        """Creates a new CelebA instance.

           Args:
             batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                 is not a divider of the dataset size (``60 000`` for train, ``10 000``
                 for test) the remainder is dropped in each epoch (after shuffling).
             train_eval_size (int): Size of the train eval data set.
                 Defaults to ``10 000`` the size of the test set.
           """
        self._name = "celeba"
        self._resize_images = resize_images
        self._train_eval_size = train_eval_size
        super(celeba, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):

        if self._resize_images:
            transform = transform_images_resize
        else:
            transform = transform_images_no_resize

        train_dataset = datasets.ImageFolder(
            root=os.path.join(config.get_data_dir(), "celeba"),
            transform=transform,
        )
        valid_dataset = datasets.ImageFolder(
            root=os.path.join(config.get_data_dir(), "celeba"),
            transform=transform,
        )
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
            train_dataset, valid_dataset
        )
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        if self._resize_images:
            transform = transform_images_resize
        else:
            transform = transform_images_no_resize

        test_dataset = datasets.ImageFolder(
            root=os.path.join(config.get_data_dir(), "celeba"),
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)
