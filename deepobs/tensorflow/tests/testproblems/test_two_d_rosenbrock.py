# -*- coding: utf-8 -*-
"""Tests for the noisy 2D Rosenbrock loss function."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import testproblems


class Two_d_RosenbrockTest(unittest.TestCase):
    """Tests for the noisy 2D Rosenbrock loss function."""

    def setUp(self):
        """Sets up the 2D dataset for the tests."""
        self.batch_size = 100
        self.two_d_rosenbrock = testproblems.two_d_rosenbrock(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.two_d_rosenbrock.set_up()
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
                1, 1
            ])
            for init_op in [
                    self.two_d_rosenbrock.train_init_op,
                    self.two_d_rosenbrock.test_init_op,
                    self.two_d_rosenbrock.train_eval_init_op
            ]:
                sess.run(init_op)
                losses_, regularizer_ = sess.run([
                    self.two_d_rosenbrock.losses, self.two_d_rosenbrock.regularizer
                ])
                self.assertEqual(losses_.shape, (self.batch_size, ))
                self.assertIsInstance(regularizer_, np.float32)


if __name__ == "__main__":
    unittest.main()
