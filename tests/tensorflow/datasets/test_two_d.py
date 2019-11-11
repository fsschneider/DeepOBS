# -*- coding: utf-8 -*-
"""Tests for the 2D dataset."""

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


class Two_dTest(unittest.TestCase):
    """Tests for the 2D dataset."""

    def setUp(self):
        """Sets up 2D dataset for the tests."""
        self.batch_size = 100
        self.two_d = datasets.two_d(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.two_d.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                self.two_d.train_init_op,
                self.two_d.test_init_op,
                self.two_d.valid_init_op,
                self.two_d.train_eval_init_op,
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.two_d.batch)
                self.assertEqual(x_.shape, (self.batch_size,))
                self.assertEqual(y_.shape, (self.batch_size,))

    def test_data_set_sizes(self):
        """Tests the sizes of the individual data sets."""
        with tf.Session() as sess:
            init_ops = [
                self.two_d.train_init_op,
                self.two_d.test_init_op,
                self.two_d.valid_init_op,
                self.two_d.train_eval_init_op,
            ]
            data_set_sizes = [
                self.two_d._train_size,
                self.two_d._train_size,
                self.two_d._train_size,
                self.two_d._train_size,
            ]
            for init_op, data_set_size in zip(init_ops, data_set_sizes):
                sess.run(init_op)
                size = 0
                while True:
                    try:
                        sess.run(self.two_d.batch)
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
