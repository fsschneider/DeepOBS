# -*- coding: utf-8 -*-
"""
This module contains utility functions for data loading in TensorFlow.
"""

import numpy as np


# We maintain a variable DATA_DIR, which contains the path to the base data
# directory. All data loading modules access dataset_utils.get_data_dir() to
# load data from this directory. The user can set it with
# deepobs.dataset_utils.set_data_dir("MYDMIR").
DATA_DIR = "data_deepobs"


def get_data_dir():
    return DATA_DIR


def set_data_dir(data_dir):
    global DATA_DIR
    DATA_DIR = data_dir


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
