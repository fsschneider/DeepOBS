# -*- coding: utf-8 -*-
"""Tolstoi DeepOBS dataset."""

import os
import numpy as np
from torch.utils import data as dat
from . import dataset
from .. import config
import torch


class tolstoi(dataset.DataSet):
    """DeepOBS data set class for character prediction on `War and Peace` by\
    Leo Tolstoi.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size the remainder is dropped in each
        epoch (after shuffling).
    seq_length (int): Sequence length to be modeled in each step.
        Defaults to ``50``.
    train_eval_size (int): Size of the train eval dataset.
        Defaults to ``653 237``, the size of the test set.

  """

    def __init__(self, batch_size, seq_length=50, train_eval_size=653237):
        """Creates a new Tolstoi instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size the remainder is dropped in each
          epoch (after shuffling).
      seq_length (int): Sequence length to be modeled in each step.
          Defaults to ``50``.
      train_eval_size (int): Size of the train eval dataset.
          Defaults to ``653 237``, the size of the test set.
    """
        self._name = "tolstoi"
        self._seq_length = seq_length
        self._train_eval_size = train_eval_size
        super(tolstoi, self).__init__(batch_size)

    def _make_dataloader(self, filepath):
        # Load the array of character ids, determine the number of batches that
        # can be produced, given batch size and sequence lengh
        arr = np.load(filepath)
        num_batches = int(
            np.floor(
                (np.size(arr) - 1) / (self._batch_size * self._seq_length)))
        if num_batches == 0:
            raise ValueError(
                "This dataset is to small to use with this batch size "
                "and sequence length.")

        # Create input and output, where output is the text shifted by one
        # character
        x = arr[:num_batches * self._batch_size * self._seq_length]
        y = arr[1:num_batches * self._batch_size * self._seq_length + 1]

#        x_sequences = x.reshape((self._batch_size  * num_batches, -1))
#        y_sequences = y.reshape((self._batch_size  * num_batches, -1))
#        dataset = dat.TensorDataset(torch.from_numpy(x_sequences), torch.from_numpy(y_sequences))
#        loader = dat.DataLoader(dataset=dataset, batch_size=self._batch_size, shuffle=False, sampler = sampler)

        # Split into batches and put into arrays X, Y, such that X[i,:] is the
        # i-th batch
        x_batches = np.split(x.reshape(self._batch_size, -1), num_batches, 1)
        y_batches = np.split(y.reshape(self._batch_size, -1), num_batches, 1)

        X = np.array(x_batches)
        Y = np.array(y_batches)

        dataset = dat.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))

        return dataset

    def _make_train_dataloader(self):
        filepath = os.path.join(config.get_data_dir(), "tolstoi", "train.npy")
        return self._make_dataloader(filepath)

    def _make_train_eval_dataloader(self):
        indices = np.arange(self._train_eval_size // (self._batch_size*self._seq_length))
        train_eval_set = self._train_dataloader[indices]
        return dat.TensorDataset(train_eval_set[0], train_eval_set[1])

    def _make_test_dataloader(self):
        filepath = os.path.join(config.get_data_dir(), "tolstoi", "test.npy")
        return self._make_dataloader(filepath)