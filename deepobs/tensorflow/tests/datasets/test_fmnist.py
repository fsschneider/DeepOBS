# -*- coding: utf-8 -*-
"""Tests for the MNIST dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import datasets


class FMNISTTest(unittest.TestCase):
    """Tests for the Fashion-MNIST dataset."""

    def setUp(self):
        """Sets up Fashion-MNIST dataset for the tests."""
        self.batch_size = 100
        self.fmnist = datasets.fmnist(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.fmnist.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                    self.fmnist.train_init_op, self.fmnist.test_init_op,
                    self.fmnist.train_eval_init_op
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.fmnist.batch)
                self.assertEqual(x_.shape, (self.batch_size, 28, 28, 1))
                self.assertEqual(y_.shape, (self.batch_size, 10))
                self.assertTrue(
                    np.allclose(np.sum(y_, axis=1), np.ones(self.batch_size)))


if __name__ == "__main__":
    unittest.main()
