# -*- coding: utf-8 -*-
"""Tests for the MLP architecture on the MNIST dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import testproblems


class MNIST_MLPTest(unittest.TestCase):
    """Test for the MLP architecture on the MNIST dataset."""

    def setUp(self):
        """Sets up MNIST dataset for the tests."""
        self.batch_size = 100
        self.mnist_mlp = testproblems.mnist_mlp(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.mnist_mlp.set_up()
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
                28 * 28 * 1000, 1000, 1000 * 500, 500, 500 * 100, 100,
                100 * 10, 10
            ])
            for init_op in [
                    self.mnist_mlp.train_init_op, self.mnist_mlp.test_init_op,
                    self.mnist_mlp.train_eval_init_op
            ]:
                sess.run(init_op)
                losses_, regularizer_, accuracy_ = sess.run([
                    self.mnist_mlp.losses, self.mnist_mlp.regularizer,
                    self.mnist_mlp.accuracy
                ])
                self.assertEqual(losses_.shape, (self.batch_size, ))
                self.assertIsInstance(regularizer_, np.float32)
                self.assertIsInstance(accuracy_, np.float32)


if __name__ == "__main__":
    unittest.main()
