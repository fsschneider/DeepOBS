# -*- coding: utf-8 -*-
"""Tests for the MNIST dataset."""

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


class MNISTTest(unittest.TestCase):
    """Tests for the MNIST dataset."""

    def setUp(self):
        """Sets up MNIST dataset for the tests."""
        self.batch_size = 100
        self.mnist = datasets.mnist(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.mnist.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all four initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                self.mnist.train_init_op,
                self.mnist.test_init_op,
                self.mnist.valid_init_op,
                self.mnist.train_eval_init_op,
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.mnist.batch)
                self.assertEqual(x_.shape, (self.batch_size, 28, 28, 1))
                self.assertEqual(y_.shape, (self.batch_size, 10))
                self.assertTrue(
                    np.allclose(np.sum(y_, axis=1), np.ones(self.batch_size))
                )

    def test_data_set_sizes(self):
        """Tests the sizes of the individual data sets."""
        with tf.Session() as sess:
            init_ops = [
                self.mnist.train_init_op,
                self.mnist.test_init_op,
                self.mnist.valid_init_op,
                self.mnist.train_eval_init_op,
            ]
            data_set_sizes = [
                60000 - self.mnist._train_eval_size,
                self.mnist._train_eval_size,
                self.mnist._train_eval_size,
                self.mnist._train_eval_size,
            ]
            for init_op, data_set_size in zip(init_ops, data_set_sizes):
                sess.run(init_op)
                size = 0
                while True:
                    try:
                        sess.run(self.mnist.batch)
                        size += 1
                    except tf.errors.OutOfRangeError:
                        print(
                            "Data set size for",
                            init_op.name,
                            ":",
                            size * self.batch_size,
                        )
                        self.assertEqual(size, data_set_size // self.batch_size)
                        break


if __name__ == "__main__":
    unittest.main()
