# -*- coding: utf-8 -*-
"""
This module contains TensorFlow data loading functionality for a Simple N-Dimensional Noisy Quadratic Problem

        0.5* (theta - x)^T * Q * (theta - x)

where x is normally distributed with mean 0.0 and sigma given by the noise_level (default is 3.0).
"""

import tensorflow as tf
import numpy as np


class data_loading:
    def __init__(self, batch_size, dim=100, train_size=1000, noise_level=6):
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
        return self.X, self.y, self.phase

    def train_dataset(self, batch_size, dim, size, noise_level):
        np.random.seed(42)
        data_x, data_y = np.random.normal(0.0, noise_level, (size, dim)), np.random.normal(0.0, noise_level, (size, dim))
        data_x = np.float32(data_x)
        data_y = np.float32(data_y)
        return self._make_dataset(data_x, data_y, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=self.train_size, num_prefetched_batches=10)

    def train_eval_dataset(self, batch_size, dim, size, noise_level):
        np.random.seed(42)
        data_x, data_y = np.random.normal(0.0, noise_level, (size, dim)), np.random.normal(0.0, noise_level, (size, dim))
        data_x = np.float32(data_x)
        data_y = np.float32(data_y)
        return self._make_dataset(data_x, data_y, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=self.train_size, num_prefetched_batches=10)

    def test_dataset(self, batch_size, dim):
        # recovers the deterministic quadratic function
        data_x, data_y = np.zeros((1, dim)), np.zeros((1, dim))
        data_x = np.float32(data_x)
        data_y = np.float32(data_y)
        return self._make_dataset(data_x, data_y, batch_size, one_hot=True, shuffle=False, num_prefetched_batches=1)

    def _make_dataset(self, data_x, data_y, batch_size, one_hot=True, shuffle=True, shuffle_buffer_size=10000, num_prefetched_batches=10):
        """Produce dataset."""
        with tf.name_scope("quadratic"):
            with tf.device('/cpu:0'):
                D = tf.data.Dataset.from_tensor_slices((data_x, data_y))
                if shuffle:
                    D = D.shuffle(buffer_size=shuffle_buffer_size)
                D = D.batch(batch_size)
                D = D.prefetch(buffer_size=num_prefetched_batches)
                return D
