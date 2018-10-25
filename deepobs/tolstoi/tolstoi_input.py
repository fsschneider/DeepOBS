# -*- coding: utf-8 -*-
"""Tensorflow data-loading functionality for character-level language modelling
on Tolstoi's War and Peace.

Large parts of this code are adapted from github.com/sherjilozair/char-rnn-tensorflow."""

import numpy as np
import tensorflow as tf
import os

from .. import dataset_utils


class data_loading:
    """Class providing the data loading functionality for the Tolstoi data set.

    Args:
        batch_size (int): Batch size of the input-output pairs. No default value is given.
        seq_length (int): Sequence length to be model in each step. No default value is given.

    Attributes:
        batch_size (int): Batch size of the input-output pairs.
        seq_length (int): Sequence length to be model in each step.
        train_eval_size (int): Number of data points to evaluate during the `train eval` phase. Currently set to ``658725`` the size of the test set.
        D_train (tf.data.Dataset): The training data set.
        D_train_eval (tf.data.Dataset): The training evaluation data set. It is the same data as `D_train` but we go through it separately.
        D_test (tf.data.Dataset): The test data set.
        phase (tf.Variable): Variable to describe which phase we are currently in. Can be "train", "train_eval" or "test". The phase variable can determine the behaviour of the network, for example deactivate dropout during evaluation.
        iterator (tf.data.Iterator): A single iterator for all three data sets. We us the initialization operators (see below) to switch this iterator to the data sets.
        X (tf.Tensor): Tensor holding the input text of the tolstoi data set for character prediction. It has dimension `batch_size` x `seq_length`.
        y (tf.Tensor): Tensor holding the target text of the tolstoi data set for character prediction, i.e. the input text shifted by a single character. It has dimension `batch_size` x `seq_length`.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch. It sets the `phase` variable to "train" and initializes the iterator to the training data set.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval phase. It sets the `phase` variable to "train_eval" and initializes the iterator to the training eval data set.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase. It sets the `phase` variable to "test" and initializes the iterator to the test data set.

    """
    def __init__(self, batch_size, seq_length):
        """Initializes the data loading class.

        Args:
            batch_size (int): Batch size of the input-output pairs.
            seq_length (int): Sequence length to be model in each step.

        """
        self.train_eval_size = 658725  # The size of the test set
        self.batch_size = batch_size
        self.seq_length = seq_length
        # Load datasets
        DATA_DIR = os.path.join(dataset_utils.get_data_dir(), "tolstoi")
        self.D_train = self.make_text_dataset(os.path.join(DATA_DIR, "train.npy"), batch_size=batch_size, seq_length=seq_length)
        self.D_train_eval = self.make_text_dataset(os.path.join(DATA_DIR, "train.npy"), batch_size=batch_size, seq_length=seq_length, data_set_size=self.train_eval_size)
        self.D_test = self.make_text_dataset(os.path.join(DATA_DIR, "test.npy"), batch_size=batch_size, seq_length=seq_length)
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
        """Returns the data (`X` (input text) and `y` (output text)) and the phase variable.

        Returns:
            tupel: Tupel consisting of the input text (`X`), the output text (`y`) and the phase variable (`phase`).

        """
        return self.X, self.y, self.phase

    def make_text_dataset(self, filepath, batch_size, seq_length, num_prefetched_batches=10, data_set_size=-1):
        """Produce a TensorFlow dataset from the filepath to the preprocessed data set.

        Args:
            filepath (str): Path to the ``.npy`` file containing the data set.
            batch_size (int): Batch size of the input-output pairs.
            seq_length (int): Sequence length to be model in each step.
            num_prefetched_batches (int): Number of prefeteched batches, defaults to ``10``.
            data_set_size (int): Size of the data set to extract from the images and label files. Defaults to ``-1`` meaning that the full data set is used.

        Returns:
            tf.data.Dataset: Data set object containing the input and output pair.

        """
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
