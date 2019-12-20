# -*- coding: utf-8 -*-
"""Tests for the Wide ResNet 16-4 architecture on the SVHN dataset."""

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



class SVHN_WRN164Test(unittest.TestCase):
    """Test for the Wide ResNet 16-4 architecture on the SVHN dataset."""

    def setUp(self):
        """Sets up SVHN dataset for the tests."""
        self.batch_size = 100
        self.svhn_wrn164 = testproblems.svhn_wrn164(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.svhn_wrn164.set_up()
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
                    3 * 16 * 3 * 3,
                    16,
                    16,
                    16 * 64 * 1 * 1,
                    16 * 64 * 3 * 3,
                    64,
                    64,
                    64 * 64 * 3 * 3,
                    64,
                    64,
                    64 * 64 * 3 * 3,
                    64,
                    64,
                    64 * 64 * 3 * 3,
                    64,
                    64,
                    64 * 128 * 1 * 1,
                    64 * 128 * 3 * 3,
                    128,
                    128,
                    128 * 128 * 3 * 3,
                    128,
                    128,
                    128 * 128 * 3 * 3,
                    128,
                    128,
                    128 * 128 * 3 * 3,
                    128,
                    128,
                    128 * 256 * 1 * 1,
                    128 * 256 * 3 * 3,
                    256,
                    256,
                    256 * 256 * 3 * 3,
                    256,
                    256,
                    256 * 256 * 3 * 3,
                    256,
                    256,
                    256 * 256 * 3 * 3,
                    256,
                    256,
                    256 * 10,
                    10,
                ],
            )
            for init_op in [
                self.svhn_wrn164.train_init_op,
                self.svhn_wrn164.test_init_op,
                self.svhn_wrn164.train_eval_init_op,
            ]:
                sess.run(init_op)
                losses_, regularizer_, accuracy_ = sess.run(
                    [
                        self.svhn_wrn164.losses,
                        self.svhn_wrn164.regularizer,
                        self.svhn_wrn164.accuracy,
                    ]
                )
                self.assertEqual(losses_.shape, (self.batch_size,))
                self.assertIsInstance(regularizer_, np.float32)
                self.assertIsInstance(accuracy_, np.float32)


if __name__ == "__main__":
    unittest.main()
