# -*- coding: utf-8 -*-
"""Tests for the CIFAR-10 dataset."""

import os
import sys
import unittest

import numpy as np
import tensorflow as tf

from deepobs.tensorflow import datasets

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)



class Cifar10Test(unittest.TestCase):
    """Tests for the CIFAR-10 dataset."""

    def setUp(self):
        """Sets up CIFAR-10 dataset for the tests."""
        self.batch_size = 100
        self.cifar10 = datasets.cifar10(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.cifar10.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all four initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                self.cifar10.train_init_op,
                self.cifar10.test_init_op,
                self.cifar10.valid_init_op,
                self.cifar10.train_eval_init_op,
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.cifar10.batch)
                self.assertEqual(x_.shape, (self.batch_size, 32, 32, 3))
                self.assertEqual(y_.shape, (self.batch_size, 10))
                self.assertTrue(
                    np.allclose(np.sum(y_, axis=1), np.ones(self.batch_size))
                )

    def test_data_set_sizes(self):
        """Tests the sizes of the individual data sets."""
        with tf.Session() as sess:
            init_ops = [
                self.cifar10.train_init_op,
                self.cifar10.test_init_op,
                self.cifar10.valid_init_op,
                self.cifar10.train_eval_init_op,
            ]
            data_set_sizes = [
                50000 - self.cifar10._train_eval_size,
                self.cifar10._train_eval_size,
                self.cifar10._train_eval_size,
                self.cifar10._train_eval_size,
            ]
            for init_op, data_set_size in zip(init_ops, data_set_sizes):
                sess.run(init_op)
                size = 0
                while True:
                    try:
                        sess.run(self.cifar10.batch)
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
