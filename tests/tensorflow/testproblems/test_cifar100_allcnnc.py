# -*- coding: utf-8 -*-
"""Tests for the All-CNN-C architecture on the CIFAR-100 dataset."""

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



class Cifar100_AllCNNCTest(unittest.TestCase):
    """Test for the All-CNN-C architecture on the CIFAR-100 dataset."""

    def setUp(self):
        """Sets up CIFAR-100 dataset for the tests."""
        self.batch_size = 100
        self.cifar100_allcnnc = testproblems.cifar100_allcnnc(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.cifar100_allcnnc.set_up()
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
                    3 * 96 * 3 * 3,
                    96,
                    96 * 96 * 3 * 3,
                    96,
                    96 * 96 * 3 * 3,
                    96,
                    96 * 192 * 3 * 3,
                    192,
                    192 * 192 * 3 * 3,
                    192,
                    192 * 192 * 3 * 3,
                    192,
                    192 * 192 * 3 * 3,
                    192,
                    192 * 192 * 1 * 1,
                    192,
                    192 * 100 * 1 * 1,
                    100,
                ],
            )
            for init_op in [
                self.cifar100_allcnnc.train_init_op,
                self.cifar100_allcnnc.test_init_op,
                self.cifar100_allcnnc.train_eval_init_op,
            ]:
                sess.run(init_op)
                losses_, regularizer_, accuracy_ = sess.run(
                    [
                        self.cifar100_allcnnc.losses,
                        self.cifar100_allcnnc.regularizer,
                        self.cifar100_allcnnc.accuracy,
                    ]
                )
                self.assertEqual(losses_.shape, (self.batch_size,))
                self.assertIsInstance(regularizer_, np.float32)
                self.assertIsInstance(accuracy_, np.float32)


if __name__ == "__main__":
    unittest.main()
