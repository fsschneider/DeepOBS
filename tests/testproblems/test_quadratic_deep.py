# -*- coding: utf-8 -*-
"""Tests for the deep quadratic loss function."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import testproblems


class Quadratic_DeepTest(unittest.TestCase):
    """Tests for the deep quadratic loss function."""

    def setUp(self):
        """Sets up the quadratic dataset for the tests."""
        self.batch_size = 10
        self.quadratic_deep = testproblems.quadratic_deep(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.quadratic_deep.set_up()
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
                100
            ])
            for init_op in [
                    self.quadratic_deep.train_init_op,
                    self.quadratic_deep.test_init_op,
                    self.quadratic_deep.train_eval_init_op
            ]:
                sess.run(init_op)
                losses_, regularizer_ = sess.run([
                    self.quadratic_deep.losses, self.quadratic_deep.regularizer
                ])
                self.assertEqual(losses_.shape, (self.batch_size, ))
                self.assertIsInstance(regularizer_, np.float32)

    def test_repeatability(self):

        quadratic_deep_1 = testproblems.quadratic_deep(batch_size = 1)
        quadratic_deep_2 = testproblems.quadratic_deep(batch_size = 1)

        np.testing.assert_almost_equal(quadratic_deep_1._hessian, quadratic_deep_2._hessian, decimal=5)

if __name__ == "__main__":
    unittest.main()
