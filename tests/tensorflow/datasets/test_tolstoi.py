# -*- coding: utf-8 -*-
"""Tests for the Tolstoi dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from deepobs.tensorflow import datasets


class TolstoiTest(unittest.TestCase):
    """Tests for the Tolstoi dataset."""

    def setUp(self):
        """Sets up Tolstoi dataset for the tests."""
        self.batch_size = 50
        self.tolstoi = datasets.tolstoi(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.tolstoi.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                self.tolstoi.train_init_op,
                self.tolstoi.test_init_op,
                self.tolstoi.valid_init_op,
                self.tolstoi.train_eval_init_op,
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.tolstoi.batch)
                self.assertEqual(
                    x_.shape, (self.batch_size, self.tolstoi._seq_length)
                )
                self.assertEqual(
                    y_.shape, (self.batch_size, self.tolstoi._seq_length)
                )
                #  check if y is x shifted by one
                x_next, _ = sess.run(self.tolstoi.batch)
                x_shift = np.roll(x_, -1, axis=1)
                x_shift[:, -1] = x_next[:, 0]
                self.assertTrue(np.allclose(x_shift, y_))

    def test_data_set_sizes(self):
        """Tests the sizes of the individual data sets.

        Note that for the Tolstoi data set, we take the full data set with size
        3,266,185 and split the test and validation set of size 653,237 off.
        """
        with tf.Session() as sess:
            init_ops = [
                self.tolstoi.train_init_op,
                self.tolstoi.test_init_op,
                self.tolstoi.valid_init_op,
                self.tolstoi.train_eval_init_op,
            ]
            data_set_sizes = [
                3266185 - 2 * self.tolstoi._train_eval_size,
                self.tolstoi._train_eval_size,
                self.tolstoi._train_eval_size,
                self.tolstoi._train_eval_size,
            ]
            for init_op, data_set_size in zip(init_ops, data_set_sizes):
                sess.run(init_op)
                size = 0
                while True:
                    try:
                        sess.run(self.tolstoi.batch)
                        size += 1
                    except tf.errors.OutOfRangeError:
                        # data_set_size is given as the number of characters,
                        # and we first have to transform it into the number of
                        # batches (where in each batch we have multiple sequences
                        # and each sequence is a collection of characters)
                        data_set_size = int(
                            np.floor(
                                (data_set_size - 1)
                                / (self.batch_size * self.tolstoi._seq_length)
                            )
                        )
                        print(
                            "Number of Batches/Sequences for",
                            init_op.name,
                            ":",
                            size,
                            size * self.batch_size,
                            "Should be",
                            data_set_size,
                            data_set_size * self.batch_size,
                        )
                        self.assertEqual(size, data_set_size)
                        break


if __name__ == "__main__":
    unittest.main()
