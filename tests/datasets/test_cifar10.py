# -*- coding: utf-8 -*-
"""Tests for the CIFAR-10 dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import datasets


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
        """Tests all three initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                    self.cifar10.train_init_op, self.cifar10.test_init_op,
                    self.cifar10.train_eval_init_op
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.cifar10.batch)
                self.assertEqual(x_.shape, (self.batch_size, 32, 32, 3))
                self.assertEqual(y_.shape, (self.batch_size, 10))
                self.assertTrue(
                    np.allclose(np.sum(y_, axis=1), np.ones(self.batch_size)))


if __name__ == "__main__":
    unittest.main()
