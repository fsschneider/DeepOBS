# -*- coding: utf-8 -*-
"""Tests for the VGG19 architecture on the CIFAR-100 dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import testproblems


class Cifar100_VGG19Test(unittest.TestCase):
    """Test for the VGG19 architecture on the CIFAR-100 dataset."""

    def setUp(self):
        """Sets up CIFAR-100 dataset for the tests."""
        self.batch_size = 100
        self.cifar100_vgg19 = testproblems.cifar100_vgg19(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.cifar100_vgg19.set_up()
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
                3 * 64 * 3 * 3, 64, 64 * 64 * 3 * 3, 64, 64 * 128 * 3 * 3, 128,
                128 * 128 * 3 * 3, 128, 128 * 256 * 3 * 3, 256,
                256 * 256 * 3 * 3, 256, 256 * 256 * 3 * 3, 256,
                256 * 256 * 3 * 3, 256, 256 * 512 * 3 * 3, 512,
                512 * 512 * 3 * 3, 512, 512 * 512 * 3 * 3, 512,
                512 * 512 * 3 * 3, 512, 512 * 512 * 3 * 3, 512,
                512 * 512 * 3 * 3, 512, 512 * 512 * 3 * 3, 512,
                512 * 512 * 3 * 3, 512, 512 * 7 * 7 * 4096, 4096, 4096 * 4096,
                4096, 4096 * 100, 100
            ])
            for init_op in [
                    self.cifar100_vgg19.train_init_op,
                    self.cifar100_vgg19.test_init_op,
                    self.cifar100_vgg19.train_eval_init_op
            ]:
                sess.run(init_op)
                losses_, regularizer_, accuracy_ = sess.run([
                    self.cifar100_vgg19.losses, self.cifar100_vgg19.regularizer,
                    self.cifar100_vgg19.accuracy
                ])
                self.assertEqual(losses_.shape, (self.batch_size, ))
                self.assertIsInstance(regularizer_, np.float32)
                self.assertIsInstance(accuracy_, np.float32)


if __name__ == "__main__":
    unittest.main()
