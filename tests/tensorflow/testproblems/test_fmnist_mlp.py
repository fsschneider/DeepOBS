# -*- coding: utf-8 -*-
"""Tests for the MLP architecture on the Fashion-MNIST dataset."""

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



class FMNIST_MLPTest(unittest.TestCase):
    """Test for the MLP architecture on the Fashion-MNIST dataset."""

    def setUp(self):
        """Sets up Fashion-MNIST dataset for the tests."""
        self.batch_size = 100
        self.fmnist_mlp = testproblems.fmnist_mlp(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.fmnist_mlp.set_up()
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
                    28 * 28 * 1000,
                    1000,
                    1000 * 500,
                    500,
                    500 * 100,
                    100,
                    100 * 10,
                    10,
                ],
            )
            for init_op in [
                self.fmnist_mlp.train_init_op,
                self.fmnist_mlp.test_init_op,
                self.fmnist_mlp.train_eval_init_op,
            ]:
                sess.run(init_op)
                losses_, regularizer_, accuracy_ = sess.run(
                    [
                        self.fmnist_mlp.losses,
                        self.fmnist_mlp.regularizer,
                        self.fmnist_mlp.accuracy,
                    ]
                )
                self.assertEqual(losses_.shape, (self.batch_size,))
                self.assertIsInstance(regularizer_, np.float32)
                self.assertIsInstance(accuracy_, np.float32)


if __name__ == "__main__":
    unittest.main()
