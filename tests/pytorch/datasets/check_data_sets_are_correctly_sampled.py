# -*- coding: utf-8 -*-
"""Script to visualize images from DeepOBS datasets."""

import os
import sys
import torch
from deepobs.pytorch import datasets
import deepobs.pytorch.config as config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
config.set_data_dir('/home/isenach/Desktop/Project/deepobs/data_deepobs')


def _check_non_labeled_dataset(dataset):
    # data augmentation must be false to check the train eval set properly
    try:
        data = dataset(batch_size = 1, data_augmentation=False)
    except TypeError:
        data = dataset(batch_size=1)

    train_loader = data._train_dataloader
    valid_loader = data._valid_dataloader
    train_eval_loader = data._train_eval_dataloader
    test_loader = data._test_dataloader

    # check sizes of sets
    assert len(train_eval_loader) == len(test_loader) == len(valid_loader)

    # check that train eval set is real subsample of train set (in not augmented case)
    iterator = iter(train_eval_loader)
    train_eval_img = iterator.next()
    assert any([torch.allclose(train_eval_img[0], train_img[0]) for train_img in train_loader])

    # check that valid set is not the same as test set
    iterator = iter(valid_loader)
    valid_img = iterator.next()
    assert not any([torch.allclose(valid_img[0], test_img[0]) for test_img in test_loader])

    # check that valid set is not a subset of the train set
    iterator = iter(valid_loader)
    valid_img = iterator.next()
    assert not any([torch.allclose(valid_img[0], train_img[0]) for train_img in train_loader])


def _check_data_set(dataset):
    # data augmentation must be false to check the train eval set properly
    try:
        data = dataset(batch_size = 1, data_augmentation=False)
    except TypeError:
        data = dataset(batch_size=1)

    train_loader = data._train_dataloader
    valid_loader = data._valid_dataloader
    train_eval_loader = data._train_eval_dataloader
    test_loader = data._test_dataloader

    # check sizes of sets
    assert len(train_eval_loader) == len(test_loader) == len(valid_loader)

    # check that train eval set is real subsample of train set (in not augmented case)
    iterator = iter(train_eval_loader)
    train_eval_img, _ = iterator.next()
    assert any([torch.allclose(train_eval_img[0], train_img[0]) for train_img, _ in train_loader])

    # check that valid set is not the same as test set
    iterator = iter(valid_loader)
    valid_img, _ = iterator.next()
    assert not any([torch.allclose(valid_img[0], test_img[0]) for test_img, _ in test_loader])

    # check that valid set is not a subset of the train set
    iterator = iter(valid_loader)
    valid_img, _ = iterator.next()
    assert not any([torch.allclose(valid_img[0], train_img[0]) for train_img, _ in train_loader])


if __name__ == "__main__":
    # _check_data_set(datasets.cifar10)
    # _check_data_set(datasets.cifar100)
    # _check_data_set(datasets.fmnist)
    # _check_data_set(datasets.mnist)
    _check_data_set(datasets.svhn)
    # _check_non_labeled_dataset(datasets.quadratic)

