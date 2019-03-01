# -*- coding: utf-8 -*-
"""Tests for the Quadratic dataset."""

import os
import sys
import unittest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
                    self.quadratic.train_init_op, self.quadratic.test_init_op,
                    self.quadratic.train_eval_init_op
            ]:
                sess.run(init_op)
                x_ = sess.run(self.quadratic.batch)
                self.assertEqual(x_.shape, (self.batch_size, self.quadratic._dim))


if __name__ == "__main__":
    unittest.main()
