# -*- coding: utf-8 -*-
"""AFHQ DeepOBS dataset."""
import os

from torchvision import datasets, transforms

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


class afhq(dataset.DataSet):
    """DeepOBS data set class for the Animal Faces-HQ`\
    <https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``14 633`` for train, ``1503``
        for test) the remainder is dropped in each epoch (after shuffling).
    train_eval_size (int): Size of the train eval dataset.
        Defaults to ``10 000``.
  """

    def __init__(self, batch_size, resize_images=False, train_eval_size=10000):
        """Creates a new CelebA instance.

           Args:
             batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
                 is not a divider of the dataset size (``14 633`` for train, ``10 000``
                 for test) the remainder is dropped in each epoch (after shuffling).
             train_eval_size (int): Size of the train eval data set.
                 Defaults to ``10 000``.
           """
        self._name = "afhq"
        self._resize_images = resize_images
        self._train_eval_size = train_eval_size
        super(afhq, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        if self._resize_images:
            transform = transform_images_resize
        else:
            transform = transform_images_no_resize

        train_dataset = datasets.ImageFolder(
            root=os.path.join(config.get_data_dir(), "afhq"),
            transform=transform,
        )
        valid_dataset = datasets.ImageFolder(
            root=os.path.join(config.get_data_dir(), "afhq"),
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
            root=os.path.join(config.get_data_dir(), "afhq"),
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None)
