# -*- coding: utf-8 -*-
"""
This module contains TensorFlow data loading functionality for Fashion-MNIST.
This is copied (with slight adaptation) from the mnist_input.py module.
"""

import tensorflow as tf
import numpy as np
import gzip
import os

from .. import dataset_utils


class data_loading:
    """Class providing the data loading functionality for the Fashion-MNIST data set.

    Args:
        batch_size (int): Batch size of the input-output pairs. No default value is given.

    Attributes:
        train_eval_size (int): Number of data points to evaluate during the `train eval` phase. Currently set to ``10000`` the size of the test set.
        D_train (tf.data.Dataset): The training data set.
        D_train_eval (tf.data.Dataset): The training evaluation data set. It is the same data as `D_train` but we go through it separately.
        D_test (tf.data.Dataset): The test data set.
        phase (tf.Variable): Variable to describe which phase we are currently in. Can be "train", "train_eval" or "test". The phase variable can determine the behaviour of the network, for example deactivate dropout during evaluation.
        iterator (tf.data.Iterator): A single iterator for all three data sets. We us the initialization operators (see below) to switch this iterator to the data sets.
        X (tf.Tensor): Tensor holding the Fashion-MNIST images. It has dimension `batch_size` x ``28`` (image size) x ``28`` (image size) x ``1`` (rgb).
        y (tf.Tensor): Label of the Fashion-MNIST images. It has dimension `batch_size` x ``10`` (number of classes).
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch. It sets the `phase` variable to "train" and initializes the iterator to the training data set.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval phase. It sets the `phase` variable to "train_eval" and initializes the iterator to the training eval data set.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase. It sets the `phase` variable to "test" and initializes the iterator to the test data set.

    """
    def __init__(self, batch_size):
        """Initializes the data loading class.

        Args:
            batch_size (int): Batch size of the input-output pairs. No default value is given.

        """
        self.train_eval_size = 10000  # The size of the test set
        self.batch_size = batch_size
        self.D_train = self.train_dataset(batch_size)
        self.D_train_eval = self.train_eval_dataset(batch_size)
        self.D_test = self.test_dataset(batch_size)
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
        """Returns the data (`X` (images) and `y` (labels)) and the phase variable.

        Returns:
            tupel: Tupel consisting of the images (`X`), the label (`y`) and the phase variable (`phase`).

        """
        return self.X, self.y, self.phase

    def train_dataset(self, batch_size):
        """Creates the training data set.

        Args:
            batch_size (int): Batch size of the input-output pairs.

        Returns:
            tf.data.Dataset: The training data set.

        """
        data_dir = dataset_utils.get_data_dir()
        train_images_file = os.path.join(
            data_dir, "fmnist", "train-images-idx3-ubyte.gz")
        train_labels_file = os.path.join(
            data_dir, "fmnist", "train-labels-idx1-ubyte.gz")
        return self.make_dataset(train_images_file, train_labels_file, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=10000, num_prefetched_batches=10)

    def train_eval_dataset(self, batch_size):
        """Creates the train eval data set.

        Args:
            batch_size (int): Batch size of the input-output pairs.

        Returns:
            tf.data.Dataset: The train eval data set.

        """
        data_dir = dataset_utils.get_data_dir()
        train_images_file = os.path.join(
            data_dir, "fmnist", "train-images-idx3-ubyte.gz")
        train_labels_file = os.path.join(
            data_dir, "fmnist", "train-labels-idx1-ubyte.gz")
        return self.make_dataset(train_images_file, train_labels_file, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=60000, num_prefetched_batches=5, data_set_size=self.train_eval_size)

    def test_dataset(self, batch_size):
        """Creates the test data set.

        Args:
            batch_size (int): Batch size of the input-output pairs.

        Returns:
            tf.data.Dataset: The test data set.

        """
        data_dir = dataset_utils.get_data_dir()
        test_images_file = os.path.join(
            data_dir, "fmnist", "t10k-images-idx3-ubyte.gz")
        test_labels_file = os.path.join(
            data_dir, "fmnist", "t10k-labels-idx1-ubyte.gz")
        return self.make_dataset(test_images_file, test_labels_file, batch_size, one_hot=True, shuffle=False, shuffle_buffer_size=-1, num_prefetched_batches=5)

    def make_dataset(self, images_file, labels_file, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=10000, num_prefetched_batches=10, data_set_size=-1):
        """Creates a data set from given images and label files.

        Args:
            images_file (str): Path to the images in compressed ``.gz`` files.
            labels_file (str): Path to the labels in compressed ``.gz`` files.
            batch_size (int): Batch size of the input-output pairs.
            one_hot (bool): Switch to turn on or off one-hot encoding of the labels. Defaults to ``True``.
            shuffle (bool):  Switch to turn on or off shuffling of the data set. Defaults to ``True``.
            shuffle_buffer_size (int): Size of the shuffle buffer. Defaults to ``10000`` the size of the `test` and `train eval` data set, meaning that they will be completely shuffled.
            num_prefetched_batches (int): Number of prefeteched batches, defaults to ``10``.
            data_set_size (int): Size of the data set to extract from the images and label files. Defaults to ``-1`` meaning that the full data set is used.

        Returns:
            tf.data.Dataset: Data set object created from the images and label files.

        """
        X, y = self.read_fmnist_data(images_file, labels_file, one_hot)
        with tf.name_scope("fmnist"):
            with tf.device('/cpu:0'):
                D = tf.data.Dataset.from_tensor_slices((X, y))
                if shuffle:
                    D = D.shuffle(buffer_size=shuffle_buffer_size)
                D = D.take(data_set_size)
                D = D.batch(batch_size)
                D = D.prefetch(buffer_size=num_prefetched_batches)
                return D

    def read32(self, bytestream):
        """Helper function to read a bytestream.

        Args:
            bytestream (bytestream): Input bytestream.

        Returns:
            np.array: Bytestream as a np array.

        """
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def read_fmnist_data(self, images_file, labels_file, one_hot=True):
        """Read the Fashion-MNIST images and labels from the downloaded files.

        Args:
            images_file (str): Path to the images in compressed ``.gz`` files.
            labels_file (str): Path to the labels in compressed ``.gz`` files.
            one_hot (bool): Switch to turn on or off one-hot encoding of the labels. Defaults to ``True``.

        Returns:
            tupel: Tupel consisting of all the images (`X`) and the labels (`y`).

        """

        # Load images from images_file
        with tf.gfile.Open(images_file, 'rb') as f:
            print('Extracting %s' % f.name)
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self.read32(bytestream)
                if magic != 2051:
                    raise ValueError('Invalid magic number %d in Fashion-MNIST image file: %s' %
                                     (magic, f.name))
                num_images = self.read32(bytestream)
                rows = self.read32(bytestream)
                cols = self.read32(bytestream)
                buf = bytestream.read(rows * cols * num_images)
                data = np.frombuffer(buf, dtype=np.uint8)
                X = data.reshape(num_images, rows, cols, 1)
                X = X.astype(np.float32) / 255.0

        # Load labels from labels file
        with tf.gfile.Open(labels_file, 'rb') as f:
            print('Extracting %s' % f.name)
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self.read32(bytestream)
                if magic != 2049:
                    raise ValueError('Invalid magic number %d in Fashion-MNIST label file: %s' %
                                     (magic, f.name))
                num_items = self.read32(bytestream)
                buf = bytestream.read(num_items)
                y = np.frombuffer(buf, dtype=np.uint8)
                if one_hot:
                    y = dataset_utils.dense_to_one_hot(y, 10)
                y = y.astype(np.int32)

        return X, y
