# -*- coding: utf-8 -*-
"""Fashion-MNIST DeepOBS dataset."""

from __future__ import print_function

import gzip
import os

import numpy as np
import tensorflow as tf

from deepobs import config

from . import dataset


class fmnist(dataset.DataSet):
    """DeepOBS data set class for the `Fashion-MNIST (FMNIST)\
    <https://github.com/zalandoresearch/fashion-mnist>`_ data set.

  Args:
    batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
        is not a divider of the dataset size (``60 000`` for train, ``10 000``
        for test) the remainder is dropped in each epoch (after shuffling).
    train_eval_size (int): Size of the train eval data set.
        Defaults to ``10 000`` the size of the test set.

  Attributes:
    batch: A tuple ``(x, y)`` of tensors, yielding batches of MNIST images
        (``x`` with shape ``(batch_size, 28, 28, 1)``) and corresponding one-hot
        label vectors (``y`` with shape ``(batch_size, 10)``). Executing these
        tensors raises a ``tf.errors.OutOfRangeError`` after one epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    valid_init_op: A tensorflow operation initializing the testproblem for
        evaluating on validation data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to ``train``, ``train_eval``,
        ``valid``, or ``test``, depending on the current phase. This can be used
        by testproblems to adapt their behavior to this phase.
  """

    def __init__(self, batch_size, train_eval_size=10000):
        """Creates a new Fashion-MNIST instance.

    Args:
      batch_size (int): The mini-batch size to use. Note that, if ``batch_size``
          is not a divider of the dataset size (``60 000`` for train, ``10 000``
          for test) the remainder is dropped in each epoch (after shuffling).
      train_eval_size (int): Size of the train eval data set.
          Defaults to ``10 000`` the size of the test set.
    """
        self._name = "fmnist"
        self._train_eval_size = train_eval_size
        super(fmnist, self).__init__(batch_size)

    def _make_dataset(self, data, shuffle=True):
        """Creates a Fashion-MNIST data set (helper used by ``.make_*_dataset`` below).

    Args:
        data (tf.data.Dataset): A tf.data.Dataset with Fashion-MNIST (train or test)
            data
        shuffle (bool):  Switch to turn on or off shuffling of the data set.
            Defaults to ``True``.

    Returns:
        A tf.data.Dataset yielding batches of Fashion-MNIST data.
    """

        with tf.compat.v1.name_scope("fmnist"):
            with tf.device("/cpu:0"):
                if shuffle:
                    data = data.shuffle(buffer_size=20000)
                data = data.batch(self._batch_size, drop_remainder=True)
                data = data.prefetch(buffer_size=4)
                return data

    def _load_dataset(self, images_file, labels_file):
        """Creates a Fashion-MNIST data set (helper used by ``.make_*_dataset`` below).

    Args:
        images_file (str): Path to the images in compressed ``.gz`` files.
        labels_file (str): Path to the labels in compressed ``.gz`` files.

    Returns:
        A tf.data.Dataset yielding Fashion-MNIST data.
    """
        X, y = self._read_mnist_data(images_file, labels_file)

        with tf.compat.v1.name_scope(self._name):
            with tf.device("/cpu:0"):
                data = tf.data.Dataset.from_tensor_slices((X, y))

        return data

    def _make_train_datasets(self):
        """Creates the three Fashion-MNIST datasets stemming from the training
        part of the data set, i.e. the training set, the training
        evaluation set, and the validation set.

    Returns:
      A tf.data.Dataset instance with batches of training data.
      A tf.data.Dataset instance with batches of training eval data.
      A tf.data.Dataset instance with batches of validation data.
    """
        data_dir = config.get_data_dir()
        train_images_file = os.path.join(
            data_dir, "fmnist", "train-images-idx3-ubyte.gz"
        )
        train_labels_file = os.path.join(
            data_dir, "fmnist", "train-labels-idx1-ubyte.gz"
        )

        data = self._load_dataset(train_images_file, train_labels_file)
        valid_data = data.take(self._train_eval_size)
        train_data = data.skip(self._train_eval_size)

        train_data = self._make_dataset(train_data, shuffle=True)
        train_eval_data = train_data.take(self._train_eval_size // self._batch_size)

        valid_data = self._make_dataset(valid_data, shuffle=False)

        return train_data, train_eval_data, valid_data

    def _make_test_dataset(self):
        """Creates the Fashion-MNIST test dataset.

    Returns:
      A tf.data.Dataset instance with batches of test data.
    """
        data_dir = config.get_data_dir()
        test_images_file = os.path.join(data_dir, "fmnist", "t10k-images-idx3-ubyte.gz")
        test_labels_file = os.path.join(data_dir, "fmnist", "t10k-labels-idx1-ubyte.gz")

        test_data = self._load_dataset(test_images_file, test_labels_file)

        return self._make_dataset(test_data, shuffle=False)

    # HELPER FUNCTIONS

    def _read_mnist_data(self, images_file, labels_file):
        """Read the Fashion-MNIST images and labels from the downloaded files.

        Args:
            images_file (str): Path to the images in compressed ``.gz`` files.
            labels_file (str): Path to the labels in compressed ``.gz`` files.

        Returns:
            tupel: Tupel consisting of all the images (`X`) and the labels (`y`).

        """
        # Load images from images_file
        with tf.io.gfile.GFile(images_file, "rb") as img_file:
            print("Extracting %s" % img_file.name)
            with gzip.GzipFile(fileobj=img_file) as bytestream:
                magic = self._read32(bytestream)
                if magic != 2051:
                    raise ValueError(
                        "Invalid magic number %d in Fashion-MNIST image file: %s"
                        % (magic, img_file.name)
                    )
                num_images = self._read32(bytestream)
                rows = self._read32(bytestream)
                cols = self._read32(bytestream)
                buf = bytestream.read(rows * cols * num_images)
                data = np.frombuffer(buf, dtype=np.uint8)
                X = data.reshape(num_images, rows, cols, 1)
                X = X.astype(np.float32) / 255.0
        # Load labels from labels file
        with tf.io.gfile.GFile(labels_file, "rb") as f:
            print("Extracting %s" % f.name)
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self._read32(bytestream)
                if magic != 2049:
                    raise ValueError(
                        "Invalid magic number %d in Fashion-MNIST label file: %s"
                        % (magic, f.name)
                    )
                num_items = self._read32(bytestream)
                buf = bytestream.read(num_items)
                y = np.frombuffer(buf, dtype=np.uint8)
                y = self._dense_to_one_hot(y, 10)
                y = y.astype(np.int32)
        return X, y

    def _read32(self, bytestream):
        """Helper function to read a bytestream.

        Args:
            bytestream (bytestream): Input bytestream.

        Returns:
            np.array: Bytestream as a np array.

        """
        dtype = np.dtype(np.uint32).newbyteorder(">")
        return np.frombuffer(bytestream.read(4), dtype=dtype)[0]

    def _dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
