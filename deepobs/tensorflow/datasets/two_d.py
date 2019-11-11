# -*- coding: utf-8 -*-
"""2D DeepOBS dataset."""

import numpy as np
import tensorflow as tf
from . import dataset


class two_d(dataset.DataSet):
    """DeepOBS data set class to create two dimensional stochastic testproblems.

    This toy data set consists of a fixed number (``train_size``) of iid draws
    from two scalar zero-mean normal distributions with standard deviation
    specified by the ``noise_level``.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``1000`` for train and test) the
        remainder is dropped in each epoch (after shuffling).
    train_size (int): Size of the training data set. This will also be used as
        the train_eval and test set size. Defaults to ``10000``.
    noise_level (float): Standard deviation of the data points around the mean.
        The data points are drawn from a Gaussian distribution. Defaults to
        ``1.0``.

  Attributes:
    batch: A tuple ``(x, y)`` of tensors with random x and y that can be used to
        create a noisy two dimensional testproblem. Executing these
        tensors raises a ``tf.errors.OutOfRangeError`` after one epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to "train", "train_eval" or
        "test", depending on the current phase. This can be used by testproblems
        to adapt their behavior to this phase.
  """

    def __init__(self, batch_size, train_size=10000, noise_level=1.0):
        """Creates a new 2D instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (1k for train and test) the
          remainder is dropped in each epoch (after shuffling).
      train_size (int): Size of the training data set. This will also be used as
          the train_eval and test set size. Defaults to ``10000``.
      noise_level (float): Standard deviation of the data points around the mean.
          The data points are drawn from a Gaussian distribution. Defaults to
          ``1.0``.
    """
        self._name = "two_d"
        self._train_size = train_size
        self._noise_level = noise_level
        super(two_d, self).__init__(batch_size)

    def _make_dataset(self, data_x, data_y, shuffle=True):
        """Creates a 2D data set (helper used by ``.make_*_datset`` below).

        Args:
          data_x (np.array): Numpy array containing the ``X`` values of the
            data points.
          data_y (np.array): Numpy array containing the ``y`` values of the
            data points.
          shuffle (bool): Switch to turn on or off shuffling of the data set.
            Defaults to ``True``.

        Returns:
            A tf.data.Dataset yielding batches of 2D data.
        """
        with tf.name_scope(self._name):
            with tf.device("/cpu:0"):
                data = tf.data.Dataset.from_tensor_slices((data_x, data_y))
                if shuffle:
                    data = data.shuffle(buffer_size=20000)
                data = data.batch(self._batch_size, drop_remainder=True)
                data = data.prefetch(buffer_size=4)
                return data

    def _make_train_datasets(self):
        """Creates the three 2D datasets stemming from the training
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
        train_x = rng.normal(0.0, self._noise_level, self._train_size)
        train_y = rng.normal(0.0, self._noise_level, self._train_size)
        train_x = np.float32(train_x)
        train_y = np.float32(train_y)
        train_data = self._make_dataset(train_x, train_y, shuffle=True)

        train_eval_data = train_data.take(self._train_size // self._batch_size)

        # Draw data from a random generator with a fixed seed to always get the
        # same data.
        rng = np.random.RandomState(44)
        valid_x = rng.normal(0.0, self._noise_level, self._train_size)
        valid_y = rng.normal(0.0, self._noise_level, self._train_size)
        valid_x = np.float32(valid_x)
        valid_y = np.float32(valid_y)
        valid_data = self._make_dataset(valid_x, valid_y, shuffle=False)

        return train_data, train_eval_data, valid_data

    def _make_test_dataset(self):
        """Creates the quadratic test dataset.

    Returns:
      A tf.data.Dataset instance with batches of test data.
    """
        # recovers the deterministic 2D function using zeros
        test_x, test_y = np.zeros(self._train_size), np.zeros(self._train_size)
        test_x = np.float32(test_x)
        test_y = np.float32(test_y)

        return self._make_dataset(test_x, test_y, shuffle=False)
