# -*- coding: utf-8 -*-
"""Quadratic DeepOBS dataset."""

import numpy as np
import tensorflow as tf
from . import dataset


class quadratic(dataset.DataSet):
    """DeepOBS data set class to create an n dimensional stochastic quadratic\
    testproblem.

    This toy data set consists of a fixed number (``train_size``) of iid draws
    from a zero-mean normal distribution in ``dim`` dimensions with isotropic
    covariance specified by ``noise_level``.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``1000`` for train and test) the
        remainder is dropped in each epoch (after shuffling).
    dim (int): Dimensionality of the quadratic. Defaults to ``100``.
    train_size (int): Size of the dataset; will be used for train, train eval and
        test datasets. Defaults to ``1000``.
    noise_level (float): Standard deviation of the data points around the mean.
        The data points are drawn from a Gaussian distribution.
        Defaults to ``0.6``.

  Attributes:
    batch: A tensor ``X`` of shape ``(batch_size, dim)`` yielding elements from
        the dataset. Executing these tensors raises a ``tf.errors.OutOfRangeError``
        after one epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to ``train``, ``train_eval``
        or ``test``, depending on the current phase. This can be used by testproblems
        to adapt their behavior to this phase.
  """

    def __init__(self, batch_size, dim=100, train_size=1000, noise_level=0.6):
        """Creates a new Quadratic instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``1000`` for train and test) the
          remainder is dropped in each epoch (after shuffling).
      dim (int): Dimensionality of the quadratic. Defaults to ``100``.
      train_size (int): Size of the dataset; will be used for train, train eval
          and test datasets. Defaults to ``1000``.
      noise_level (float): Standard deviation of the data points around the mean.
          The data points are drawn from a Gaussian distribution.
          Defaults to ``0.6``.
    """
        self._name = "quadratic"
        self._dim = dim
        self._train_size = train_size
        self._noise_level = noise_level
        super(quadratic, self).__init__(batch_size)

    def _make_dataset(self, X, shuffle=True):
        """Creates a quadratic data set (helper used by ``.make_*_datset`` below).

        Args:
            X (np.array): Numpy array containing the ``x`` values of the data points.
            data_y (np.array): Numpy array containing the ``y`` values of the data points.
            shuffle (bool):  Switch to turn on or off shuffling of the data set.
                Defaults to ``True``.

        Returns:
            A tf.data.Dataset yielding batches of quadratic data.
        """
        with tf.name_scope(self._name):
            with tf.device("/cpu:0"):
                data = tf.data.Dataset.from_tensor_slices(X)
                if shuffle:
                    data = data.shuffle(buffer_size=20000)
                data = data.batch(self._batch_size, drop_remainder=True)
                data = data.prefetch(buffer_size=4)
                return data

    def _make_train_datasets(self):
        """Creates the three quadratic datasets stemming from the training
        part of the data set, i.e. the training set, the training
        evaluation set, and the validation set.

    Returns:
      A tf.data.Dataset instance with batches of training data.
      A tf.data.Dataset instance with batches of training eval data.
      A tf.data.Dataset instance with batches of validation data.
    """
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        rng = np.random.RandomState(42)
        X_train = rng.normal(
            0.0, self._noise_level, (self._train_size, self._dim)
        )
        X_train = np.float32(X_train)
        train_data = self._make_dataset(X_train, shuffle=True)

        train_eval_data = train_data.take(self._train_size // self._batch_size)

        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        rng = np.random.RandomState(44)
        X_valid = rng.normal(
            0.0, self._noise_level, (self._train_size, self._dim)
        )
        X_valid = np.float32(X_valid)
        valid_data = self._make_dataset(X_valid, shuffle=False)

        return train_data, train_eval_data, valid_data

    def _make_test_dataset(self):
        """Creates the quadratic test dataset.

    Returns:
      A tf.data.Dataset instance with batches of test data.
    """
        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        rng = np.random.RandomState(43)
        X_test = rng.normal(
            0.0, self._noise_level, (self._train_size, self._dim)
        )
        X_test = np.float32(X_test)

        return self._make_dataset(X_test, shuffle=False)

