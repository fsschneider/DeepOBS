# -*- coding: utf-8 -*-
"""Tests for the VAE on the Fashion-MNIST dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import testproblems


class FMNIST_VAETest(unittest.TestCase):
    """Test for the VAE on the Fashion-MNIST dataset."""

    def setUp(self):
        """Sets up Fashion-MNIST dataset for the tests."""
        self.batch_size = 100
        self.fmnist_vae = testproblems.fmnist_vae(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.fmnist_vae.set_up()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_param = [
                np.prod(v.get_shape().as_list())
                for v in tf.trainable_variables()
            ]
            # Check if number of parameters per "layer" is equal to what we expect
            # We will write them in the following form:
            # - Conv layer: [input_filter*output_filter*kernel[0]*kernel[1]]
            # - Batch norm: [input, input] (for beta and gamma)
            # - Fully connected: [input*output]
            # - Bias: [dim]
            self.assertEqual(num_param, [
                1 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64,
                7 * 7 * 64 * 8, 8, 7 * 7 * 64 * 8, 8, 8 * 24, 24, 24 * 49, 49,
                1 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64,
                14 * 14 * 64 * 28 * 28, 28 * 28
            ])
            for init_op in [
                    self.fmnist_vae.train_init_op,
                    self.fmnist_vae.test_init_op,
                    self.fmnist_vae.train_eval_init_op
            ]:
                sess.run(init_op)
                losses_, regularizer_ = sess.run(
                    [self.fmnist_vae.losses, self.fmnist_vae.regularizer])
                self.assertEqual(losses_.shape, (self.batch_size, ))
                self.assertIsInstance(regularizer_, np.float32)


if __name__ == "__main__":
    unittest.main()
