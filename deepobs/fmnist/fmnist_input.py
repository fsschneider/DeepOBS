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
    def __init__(self, batch_size):
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
        return self.X, self.y, self.phase

    def train_dataset(self, batch_size):
        data_dir = dataset_utils.get_data_dir()
        train_images_file = os.path.join(
            data_dir, "fmnist", "train-images-idx3-ubyte.gz")
        train_labels_file = os.path.join(
            data_dir, "fmnist", "train-labels-idx1-ubyte.gz")
        return self._make_dataset(train_images_file, train_labels_file, batch_size,
                                  one_hot=True, shuffle=True, shuffle_buffer_size=10000,
                                  num_prefetched_batches=10)

    def train_eval_dataset(self, batch_size):
        data_dir = dataset_utils.get_data_dir()
        train_images_file = os.path.join(
            data_dir, "fmnist", "train-images-idx3-ubyte.gz")
        train_labels_file = os.path.join(
            data_dir, "fmnist", "train-labels-idx1-ubyte.gz")
        return self._make_dataset(train_images_file, train_labels_file, batch_size,
                                  one_hot=True, shuffle=True, shuffle_buffer_size=60000,
                                  num_prefetched_batches=5, data_set_size=self.train_eval_size)

    def test_dataset(self, batch_size):
        data_dir = dataset_utils.get_data_dir()
        test_images_file = os.path.join(
            data_dir, "fmnist", "t10k-images-idx3-ubyte.gz")
        test_labels_file = os.path.join(
            data_dir, "fmnist", "t10k-labels-idx1-ubyte.gz")
        return self._make_dataset(test_images_file, test_labels_file, batch_size,
                                  one_hot=True, shuffle=False, shuffle_buffer_size=-1,
                                  num_prefetched_batches=5)

    def _make_dataset(self, images_file, labels_file, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=10000, num_prefetched_batches=10, data_set_size=-1):
        """Produce Fashion-MNIST dataset."""
        X, y = self._read_fmnist_data(images_file, labels_file, one_hot)
        with tf.name_scope("fmnist"):
            with tf.device('/cpu:0'):
                D = tf.data.Dataset.from_tensor_slices((X, y))
                if shuffle:
                    D = D.shuffle(buffer_size=shuffle_buffer_size)
                D = D.take(data_set_size)
                D = D.batch(batch_size)
                D = D.prefetch(buffer_size=num_prefetched_batches)
                return D

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def _read_fmnist_data(self, images_file, labels_file, one_hot=True):
        """Extract Fashion-MNIST data from the provided .gz files for images and labels.

        Arguments:
          :images_file: Path to the .gz file containing Fashion-MNIST images.
          :labels_file: Path to the .gz file containing corresponding labels.
          :one_hot: Boolean. If True, the labels are returned as one-hot vectors.

        Returns:
          :X: Numpy array of shape [num_images, 28, 28, 1] containing Fashion-MNIST images.
          :y: Numpy array containing corresponding labels. If ``one_hot=True``, its
              shape is [num_images, 10], otherwise [num_images]."""

        # Load images from images_file
        with tf.gfile.Open(images_file, 'rb') as f:
            print('Extracting %s' % f.name)
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self._read32(bytestream)
                if magic != 2051:
                    raise ValueError('Invalid magic number %d in Fashion-MNIST image file: %s' %
                                     (magic, f.name))
                num_images = self._read32(bytestream)
                rows = self._read32(bytestream)
                cols = self._read32(bytestream)
                buf = bytestream.read(rows * cols * num_images)
                data = np.frombuffer(buf, dtype=np.uint8)
                X = data.reshape(num_images, rows, cols, 1)
                X = X.astype(np.float32) / 255.0

        # Load labels from labels file
        with tf.gfile.Open(labels_file, 'rb') as f:
            print('Extracting %s' % f.name)
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self._read32(bytestream)
                if magic != 2049:
                    raise ValueError('Invalid magic number %d in Fashion-MNIST label file: %s' %
                                     (magic, f.name))
                num_items = self._read32(bytestream)
                buf = bytestream.read(num_items)
                y = np.frombuffer(buf, dtype=np.uint8)
                if one_hot:
                    y = dataset_utils.dense_to_one_hot(y, 10)
                y = y.astype(np.int32)

        return X, y
