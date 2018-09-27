# -*- coding: utf-8 -*-
"""Tensorflow data-loading functionality for character-level language modelling
on Tolstoi's War and Peace.

Large parts of this code are adapted from github.com/sherjilozair/char-rnn-tensorflow."""

import numpy as np
import tensorflow as tf
import os

from .. import dataset_utils


class data_loading:
    def __init__(self, batch_size, seq_length):
        self.train_eval_size = 658725  # The size of the test set
        self.batch_size = batch_size
        self.seq_length = seq_length
        # Load datasets
        DATA_DIR = os.path.join(dataset_utils.get_data_dir(), "tolstoi")
        self.D_train = self._make_text_dataset(os.path.join(DATA_DIR, "train.npy"), batch_size=batch_size, seq_length=seq_length)
        self.D_train_eval = self._make_text_dataset(os.path.join(DATA_DIR, "train.npy"), batch_size=batch_size, seq_length=seq_length, data_set_size=self.train_eval_size)
        self.D_test = self._make_text_dataset(os.path.join(DATA_DIR, "test.npy"), batch_size=batch_size, seq_length=seq_length)
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

    def _make_text_dataset(self, filepath, batch_size, seq_length, num_prefetched_batches=10, data_set_size=-1):
        # Load the array of character ids, determine the number of batches that can be
        # produced, given batch size and sequence lengh
        arr = np.load(filepath)
        print np.shape(arr)
        num_batches = int(np.floor((np.size(arr) - 1) / (batch_size * seq_length)))
        print "Num batches:", num_batches
        if num_batches == 0:
            raise ValueError("This dataset is to small to use with this batch size "
                             "and sequence length.")

        # Create input and output, where output is the text shifted by one character
        x = arr[:num_batches * batch_size * seq_length]
        y = arr[1:num_batches * batch_size * seq_length + 1]

        # Split into batches and put into arrays X, Y, such that X[i,:] is the i-th batch
        x_batches = np.split(x.reshape(batch_size, -1), num_batches, 1)
        y_batches = np.split(y.reshape(batch_size, -1), num_batches, 1)
        X = np.array(x_batches)
        Y = np.array(y_batches)

        # Create dataset from X, Y arrays
        D = tf.data.Dataset.from_tensor_slices((X, Y))
        D = D.take(data_set_size)
        D = D.prefetch(buffer_size=num_prefetched_batches)
        return D
