# -*- coding: utf-8 -*-
"""Tests for the 2c2d architecture on the MNIST dataset."""

import os
import sys
import unittest

import numpy as np
import tensorflow as tf

from deepobs.tensorflow import testproblems

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)



class MNIST_2c2dTest(unittest.TestCase):
    """Test for the 2c2d architecture on the MNIST dataset."""

    def setUp(self):
        """Sets up MNIST dataset for the tests."""
        self.batch_size = 100
        self.mnist_2c2d = testproblems.mnist_2c2d(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.mnist_2c2d.set_up()
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
            self.assertEqual(
                num_param,
                [
                    1 * 32 * 5 * 5,
                    32,
                    32 * 64 * 5 * 5,
                    64,
                    7 * 7 * 64 * 1024,
                    1024,
                    1024 * 10,
                    10,
                ],
            )
            for init_op in [
                self.mnist_2c2d.train_init_op,
                self.mnist_2c2d.test_init_op,
                self.mnist_2c2d.train_eval_init_op,
            ]:
                sess.run(init_op)
                losses_, regularizer_, accuracy_ = sess.run(
                    [
                        self.mnist_2c2d.losses,
                        self.mnist_2c2d.regularizer,
                        self.mnist_2c2d.accuracy,
                    ]
                )
                self.assertEqual(losses_.shape, (self.batch_size,))
                self.assertIsInstance(regularizer_, np.float32)
                self.assertIsInstance(accuracy_, np.float32)


if __name__ == "__main__":
    unittest.main()
