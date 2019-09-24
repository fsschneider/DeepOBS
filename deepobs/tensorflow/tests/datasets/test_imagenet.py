# -*- coding: utf-8 -*-
"""Tests for the ImageNet dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import datasets


class ImagenetTest(unittest.TestCase):
    """Tests for the ImageNet dataset."""

    def setUp(self):
        """Sets up ImageNet dataset for the tests."""
        self.batch_size = 1000
        self.imagenet = datasets.imagenet(self.batch_size)

    def test_phase_not_trainable(self):
        """Makes sure the ``phase`` variable is not trainable."""
        phase = self.imagenet.phase
        self.assertFalse(phase.trainable)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        with tf.Session() as sess:
            for init_op in [
                    self.imagenet.train_init_op, self.imagenet.test_init_op,
                    self.imagenet.train_eval_init_op
            ]:
                sess.run(init_op)
                x_, y_ = sess.run(self.imagenet.batch)
                self.assertEqual(x_.shape, (self.batch_size, 224, 224, 3))
                self.assertEqual(y_.shape, (self.batch_size, 1001))
                self.assertTrue(
                    np.allclose(np.sum(y_, axis=1), np.ones(self.batch_size)))


if __name__ == "__main__":
    unittest.main()
