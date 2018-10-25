# -*- coding: utf-8 -*-
"""
This module contains TensorFlow data loading functionality for a Simple N-Dimensional Noisy Quadratic Problem

        0.5* (theta - x)^T * Q * (theta - x)

where x is normally distributed with mean 0.0 and sigma given by the noise_level (default is 3.0).
"""

import tensorflow as tf
import numpy as np


class data_loading:
    """Short summary.

    Args:
        batch_size (int): Batch size. No default value is given.
        dim (int): Dimensionality of the data points and therefore the created quadratic problem. Defaults to ``100``
        train_size (int): Size of the training set. Defaults to ``1000``.
        noise_level (float): Noise level of the training set. All training points are sampled from a gaussian distribution with the noise level as the standard deviation. Defaults to ``6``.

    Attributes:
        batch_size (int): Batch size. No default value is given.
        dim (int): Dimensionality of the data points and therefore the created quadratic problem. Defaults to ``100``
        train_size (int): Size of the training set. Defaults to ``1000``.
        noise_level (float): Noise level of the training set. All training points are sampled from a gaussian distribution with the noise level as the standard deviation. Defaults to ``6``.
        D_train (tf.data.Dataset): The training data set.
        D_train_eval (tf.data.Dataset): The training evaluation data set. It is the same data as `D_train` but we go through it separately.
        D_test (tf.data.Dataset): The test data set. We use the mean of the data points. Thus, the test data set has just a single data point.
        phase (tf.Variable): Variable to describe which phase we are currently in. Can be "train", "train_eval" or "test". The phase variable can determine the behaviour of the network, for example deactivate dropout during evaluation.
        iterator (tf.data.Iterator): A single iterator for all three data sets. We us the initialization operators (see below) to switch this iterator to the data sets.
        X (tf.Tensor): Tensor holding data points. It has dimension `batch_size`.
        y (tf.Tensor): Tensor holding the labels of the data points. It has dimension `batch_size`.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch. It sets the `phase` variable to "train" and initializes the iterator to the training data set.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval phase. It sets the `phase` variable to "train_eval" and initializes the iterator to the training eval data set.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase. It sets the `phase` variable to "test" and initializes the iterator to the test data set.

    """
    def __init__(self, batch_size, dim=100, train_size=1000, noise_level=6):
        """Initializes the data loading class.

        Args:
            batch_size (int): Batch size. No default value is given.
            dim (int): Dimensionality of the data points and therefore the created quadratic problem. Defaults to ``100``
            train_size (int): Size of the training set. Defaults to ``1000``.
            noise_level (float): Noise level of the training set. All training points are sampled from a gaussian distribution with the noise level as the standard deviation. Defaults to ``6``.

        """
        self.train_size = train_size  # The size of the test set
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.dim = dim
        self.D_train = self.train_dataset(batch_size, dim=dim, size=train_size, noise_level=noise_level)
        self.D_train_eval = self.train_eval_dataset(batch_size, dim=dim, size=train_size, noise_level=noise_level)
        self.D_test = self.test_dataset(batch_size, dim=dim)
        self.phase = tf.Variable("train", name="phase", trainable=False)

        # Reinitializable iterator given types and shapes of the outputs (needs to be the same for train and test of course)
        self.iterator = tf.data.Iterator.from_structure(
            self.D_train.output_types, self.D_train.output_shapes)
        self.X, self.y = self.iterator.get_next()

        # Operations to do when switching the phase (initialize iterator and assign phase to phase variable)
        self.train_init_op = tf.group([self.iterator.make_initializer(
            self.D_train), tf.assign(self.phase, "train")], name="train_init_op")
        self.train_eval_init_op = tf.group([self.iterator.make_initializer(
            self.D_train_eval), tf.assign(self.phase, "train_eval")], name="train_eval_init_op")
        self.test_init_op = tf.group([self.iterator.make_initializer(
            self.D_test), tf.assign(self.phase, "test")], name="test_init_op")

    def load(self):
        """Returns the data (`X` and `y` ) and the phase variable.

        Returns:
            tupel: Tupel consisting of the data points (`X`), (`y`) and the phase variable (`phase`).

        """
        return self.X, self.y, self.phase

    def train_dataset(self, batch_size, dim, size, noise_level):
        """Creates the training data set.

        Args:
            batch_size (int): Batch size of the data points.
            dim (int): Dimensionality of each dat point.
            size (int): Size of the training data set, i.e. the number of data points in the train set.
            noise_level (float): Standard deviation of the data points around the mean. The data points are drawn from a Gaussian distribution.

        Returns:
            tf.data.Dataset: The training data set.

        """
        np.random.seed(42)
        data_x, data_y = np.random.normal(0.0, noise_level, (size, dim)), np.random.normal(0.0, noise_level, (size, dim))
        data_x = np.float32(data_x)
        data_y = np.float32(data_y)
        return self.make_dataset(data_x, data_y, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=self.train_size, num_prefetched_batches=10)

    def train_eval_dataset(self, batch_size, dim, size, noise_level):
        """Creates the train eval data set.

        Args:
            batch_size (int): Batch size of the data points.
            dim (int): Dimensionality of each dat point.
            size (int): Size of the train eval data set, i.e. the number of data points in the train eval set.
            noise_level (float): Standard deviation of the data points around the mean. The data points are drawn from a Gaussian distribution.

        Returns:
            tf.data.Dataset: The train eval data set.

        """
        np.random.seed(42)
        data_x, data_y = np.random.normal(0.0, noise_level, (size, dim)), np.random.normal(0.0, noise_level, (size, dim))
        data_x = np.float32(data_x)
        data_y = np.float32(data_y)
        return self.make_dataset(data_x, data_y, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=self.train_size, num_prefetched_batches=10)

    def test_dataset(self, batch_size, dim):
        """Creates the test data set.

        Args:
            batch_size (int): Batch size. Just a single data point is created, with the mean value of the Gaussian distributions of the training data set.
            dim (int): Dimension of the data points.

        Returns:
            tf.data.Dataset: The test data set.

        """
        # recovers the deterministic quadratic function
        data_x, data_y = np.zeros((1, dim)), np.zeros((1, dim))
        data_x = np.float32(data_x)
        data_y = np.float32(data_y)
        return self.make_dataset(data_x, data_y, batch_size, one_hot=True, shuffle=False, num_prefetched_batches=1)

    def make_dataset(self, data_x, data_y, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=10000, num_prefetched_batches=10):
        """Creates a data set from given data points.

        Args:
            data_x (np.array): Numpy array containing the ``X`` values of the data points.
            data_y (np.array): Numpy array containing the ``y`` values of the data points.
            batch_size (int): Batch size of the input-output pairs.
            one_hot (bool): Switch to turn on or off one-hot encoding of the labels. Defaults to ``True``.
            shuffle (bool):  Switch to turn on or off shuffling of the data set. Defaults to ``True``.
            shuffle_buffer_size (int): Size of the shuffle buffer. Defaults to ``10000`` the size of the `test` and `train eval` data set, meaning that they will be completely shuffled.
            num_prefetched_batches (int): Number of prefeteched batches, defaults to ``10``.

        Returns:
            tf.data.Dataset: Data set object created from the images and label files.

        """
        with tf.name_scope("quadratic"):
            with tf.device('/cpu:0'):
                D = tf.data.Dataset.from_tensor_slices((data_x, data_y))
                if shuffle:
                    D = D.shuffle(buffer_size=shuffle_buffer_size)
                D = D.batch(batch_size)
                D = D.prefetch(buffer_size=num_prefetched_batches)
                return D
