# -*- coding: utf-8 -*-
"""Tolstoi DeepOBS dataset."""

import os
import numpy as np
import tensorflow as tf
from . import dataset
from deepobs import config


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

  Attributes:
    batch: A tuple ``(x, y)`` of tensors, yielding batches of tolstoi data
        (``x`` with shape ``(batch_size, seq_length)``) and (``y`` with shape
        ``(batch_size, seq_length)`` which is ``x`` shifted by one).
        Executing these tensors raises a ``tf.errors.OutOfRangeError`` after one
        epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to ``train``, ``train_eval``
        or ``test``, depending on the current phase. This can be used by
        testproblems to adapt their behavior to this phase.
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

    def _make_dataset(self, arr):
        """Creates a Tolstoi data set (helper used by ``.make_*_datset`` below).

    Args:
        filepath (str): Filepath to the .npy file containing the data set.

    Returns:
        A tf.data.Dataset yielding batches of Tolstoi data.
    """
        # Load the array of character ids, determine the number of batches that
        # can be produced, given batch size and sequence length
        num_batches = int(
            np.floor((np.size(arr) - 1) / (self._batch_size * self._seq_length))
        )
        if num_batches == 0:
            raise ValueError(
                "This dataset is to small to use with this batch size "
                "and sequence length."
            )

        # Create input and output, where output is the text shifted by one
        # character
        x = arr[: num_batches * self._batch_size * self._seq_length]
        y = arr[1 : num_batches * self._batch_size * self._seq_length + 1]

        # Split into batches and put into arrays X, Y, such that X[i,:] is the
        # i-th batch
        x_batches = np.split(x.reshape(self._batch_size, -1), num_batches, 1)
        y_batches = np.split(y.reshape(self._batch_size, -1), num_batches, 1)
        X = np.array(x_batches)
        Y = np.array(y_batches)

        with tf.name_scope(self._name):
            with tf.device("/cpu:0"):
                data = tf.data.Dataset.from_tensor_slices((X, Y))
                data = data.prefetch(buffer_size=4)
                return data

    def _make_train_datasets(self):
        """Creates the three Tolstoi datasets stemming from the training
        part of the data set, i.e. the training set, the training
        evaluation set, and the validation set.

    Returns:
      A tf.data.Dataset instance with batches of training data.
      A tf.data.Dataset instance with batches of training eval data.
      A tf.data.Dataset instance with batches of validation data.
    """
        filepath = os.path.join(config.get_data_dir(), "tolstoi", "train.npy")

        data = np.load(filepath)

        valid_data = data[0 : self._train_eval_size]
        train_data = data[self._train_eval_size :]

        train_data = self._make_dataset(train_data)
        train_eval_data = train_data.take(
            self._train_eval_size // (self._batch_size * self._seq_length)
        )

        valid_data = self._make_dataset(valid_data)

        return train_data, train_eval_data, valid_data

    def _make_test_dataset(self):
        """Creates the Tolstoi test dataset.

    Returns:
      A tf.data.Dataset instance with batches of test data.
    """
        filepath = os.path.join(config.get_data_dir(), "tolstoi", "test.npy")

        data = np.load(filepath)

        return self._make_dataset(data)

