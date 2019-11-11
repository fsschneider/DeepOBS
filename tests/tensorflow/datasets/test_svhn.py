# -*- coding: utf-8 -*-
"""Tests for the SVHN dataset."""

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


class SVHNTest(unittest.TestCase):
    """Tests for the SVHN dataset."""

    def setUp(self):
        """Sets up SVHN dataset for the tests."""
        self.batch_size = 25
        self.svhn = datasets.svhn(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.svhn.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all four initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                self.svhn.train_init_op,
                self.svhn.test_init_op,
                self.svhn.valid_init_op,
                self.svhn.train_eval_init_op,
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.svhn.batch)
                self.assertEqual(x_.shape, (self.batch_size, 32, 32, 3))
                self.assertEqual(y_.shape, (self.batch_size, 10))
                self.assertTrue(
                    np.allclose(np.sum(y_, axis=1), np.ones(self.batch_size))
                )

    def test_data_set_sizes(self):
        """Tests the sizes of the individual data sets."""
        with tf.Session() as sess:
            init_ops = [
                self.svhn.train_init_op,
                self.svhn.test_init_op,
                self.svhn.valid_init_op,
                self.svhn.train_eval_init_op,
            ]
            data_set_sizes = [
                73257 - self.svhn._train_eval_size,
                self.svhn._train_eval_size,
                self.svhn._train_eval_size,
                self.svhn._train_eval_size,
            ]
            for init_op, data_set_size in zip(init_ops, data_set_sizes):
                sess.run(init_op)
                size = 0
                while True:
                    try:
                        sess.run(self.svhn.batch)
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
