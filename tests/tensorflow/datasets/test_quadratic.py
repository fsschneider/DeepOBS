# -*- coding: utf-8 -*-
"""Tests for the Quadratic dataset."""

import os
import sys
import unittest
import tensorflow as tf

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from deepobs.tensorflow import datasets


class QuadraticTest(unittest.TestCase):
    """Tests for the Quadratic dataset."""

    def setUp(self):
        """Sets up Quadratic dataset for the tests."""
        self.batch_size = 100
        self.quadratic = datasets.quadratic(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.quadratic.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                self.quadratic.train_init_op,
                self.quadratic.test_init_op,
                self.quadratic.valid_init_op,
                self.quadratic.train_eval_init_op,
            ]:
                sess.run(init_op)
                x_ = sess.run(self.quadratic.batch)
                self.assertEqual(
                    x_.shape, (self.batch_size, self.quadratic._dim)
                )

    def test_data_set_sizes(self):
        """Tests the sizes of the individual data sets."""
        with tf.Session() as sess:
            init_ops = [
                self.quadratic.train_init_op,
                self.quadratic.test_init_op,
                self.quadratic.valid_init_op,
                self.quadratic.train_eval_init_op,
            ]
            data_set_sizes = [
                self.quadratic._train_size,
                self.quadratic._train_size,
                self.quadratic._train_size,
                self.quadratic._train_size,
            ]
            for init_op, data_set_size in zip(init_ops, data_set_sizes):
                sess.run(init_op)
                size = 0
                while True:
                    try:
                        sess.run(self.quadratic.batch)
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

