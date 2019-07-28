# -*- coding: utf-8 -*-
"""Tests for the Tolstoi dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import datasets


class TolstoiTest(unittest.TestCase):
    """Tests for the Tolstoi dataset."""

    def setUp(self):
        """Sets up Tolstoi dataset for the tests."""
        self.batch_size = 100
        self.tolstoi = datasets.tolstoi(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.tolstoi.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                    self.tolstoi.train_init_op, self.tolstoi.test_init_op,
                    self.tolstoi.train_eval_init_op
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.tolstoi.batch)
                self.assertEqual(x_.shape, (self.batch_size, self.tolstoi._seq_length))
                self.assertEqual(y_.shape, (self.batch_size, self.tolstoi._seq_length))
                #  check if y is x shifted by one
                x_next, _ = sess.run(self.tolstoi.batch)
                x_shift = np.roll(x_, -1, axis=1)
                x_shift[:, -1] = x_next[:, 0]
                self.assertTrue(np.allclose(x_shift, y_))

if __name__ == "__main__":
    unittest.main()
