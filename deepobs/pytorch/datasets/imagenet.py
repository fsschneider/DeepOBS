# -*- coding: utf-8 -*-
"""ImageNet DeepOBS dataset."""

import os

from deepobs import config
from torchvision import datasets, transforms

from . import dataset

training_transform_not_augmented = transforms.Compose(
    [
        transforms.Resize(size=256),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ]
)

training_transform_augmented = transforms.Compose(
    [
        transforms.Resize(size=256),
        transforms.CenterCrop(size=(256, 256)),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=32.0 / 255.0,
            saturation=[0.5, 1.5],
            hue=0.2,
            contrast=[0.5, 1.5],
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ]
)


class imagenet(dataset.DataSet):
    """DeepOBS data set class for the `ImageNet\
    <http://www.image-net.org/>`_ data set.

    .. NOTE::
        We use ``1001`` classes  which includes an additional `background` class,
        as it is used for example by the inception net.

    Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in
            each epoch (after shuffling).
        data_augmentation (bool): If ``True`` some data augmentation operations
            (random crop window, horizontal flipping, lighting augmentation) are
            applied to the training data (but not the test data).
        train_eval_size (int): Size of the train eval data set.
            Defaults to ``50 000`` the size of the test set.

    Methods:
        _make_dataloader: A helper that is shared by all three data loader methods.
    """

    def __init__(self, batch_size, data_augmentation=True, train_eval_size=50000):
        """Create a new ImageNet instance.

        Args:
        batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
            is not a divider of the dataset size the remainder is dropped in
            each epoch (after shuffling).
        data_augmentation (bool): If ``True`` some data augmentation operations
            (random crop window, horizontal flipping, lighting augmentation) are
            applied to the training data (but not the test data).
        train_eval_size (int): Size of the train eval data set.
            Defaults to ``50 000`` the size of the test set.
        """
        self._name = "imagenet"
        self._data_augmentation = data_augmentation
        self._train_eval_size = train_eval_size
        super(imagenet, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        if self._data_augmentation:
            transform = training_transform_augmented
        else:
            transform = training_transform_not_augmented

        train_dataset = datasets.ImageNet(
            root=os.path.join(config.get_data_dir(), "imagenet/pytorch"),
            split="val",  # TODO Change to train again
            transform=transform,
        )
        valid_dataset = datasets.ImageNet(
            root=os.path.join(config.get_data_dir(), "imagenet/pytorch"),
            split="val",  # TODO Change to train again
            transform=transform,
        )
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(
            train_dataset, valid_dataset
        )
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = training_transform_not_augmented
        test_dataset = datasets.ImageNet(
            root=os.path.join(config.get_data_dir(), "imagenet/pytorch"),
            split="val",
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)
