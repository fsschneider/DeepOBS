# -*- coding: utf-8 -*-
"""Tests for the Inception v3 architecture on the ImageNet dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import testproblems


class Imagenet_Inception_v3Test(unittest.TestCase):
    """Test for the Inception v3 architecture on the ImageNet dataset."""

    def setUp(self):
        """Sets up ImageNet dataset for the tests."""
        self.batch_size = 1
        self.imagenet_inception_v3 = testproblems.imagenet_inception_v3(
            self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.imagenet_inception_v3.set_up()
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
                num_param, [
                    3 * 32 * 3 * 3, 32, 32, 32 * 32 * 3 * 3, 32, 32,
                    32 * 64 * 3 * 3, 64, 64, 64 * 80 * 1 * 1, 80, 80,
                    80 * 192 * 3 * 3, 192, 192, 192 * 64 * 1 * 1, 64, 64,
                    192 * 32 * 1 * 1, 32, 32, 192 * 48 * 1 * 1, 48, 48,
                    48 * 64 * 5 * 5, 64, 64, 192 * 64 * 1 * 1,
                    64, 64, 64 * 96 * 3 * 3, 96, 96, 96 * 96 * 3 * 3, 96, 96,
                    (64 + 32 + 64 + 96) * 64 * 1 * 1, 64, 64,
                    (64 + 32 + 64 + 96) * 64 * 1 * 1, 64, 64,
                    (64 + 32 + 64 + 96) * 48 * 1 * 1, 48, 48, 48 * 64 * 5 * 5,
                    64, 64, (64 + 32 + 64 + 96) * 64 * 1 * 1, 64, 64,
                    64 * 96 * 3 * 3, 96, 96, 96 * 96 * 3 * 3, 96, 96,
                    (64 + 64 + 64 + 96) * 64 * 1 * 1, 64, 64,
                    (64 + 64 + 64 + 96) * 64 * 1 * 1, 64, 64,
                    (64 + 64 + 64 + 96) * 48 * 1 * 1, 48, 48, 48 * 64 * 5 * 5,
                    64, 64, (64 + 64 + 64 + 96) * 64 * 1 * 1, 64, 64,
                    64 * 96 * 3 * 3, 96, 96, 96 * 96 * 3 * 3, 96, 96,
                    (64 + 64 + 64 + 96) * 384 * 3 * 3, 384, 384,
                    (64 + 64 + 64 + 96) * 64 * 1 * 1, 64, 64, 64 * 96 * 3 * 3,
                    96, 96, 96 * 96 * 3 * 3, 96, 96,
                    ((64 + 64 + 64 + 96) + 384 + 96) * 192 * 1 * 1, 192, 192,
                    ((64 + 64 + 64 + 96) + 384 + 96) * 192 * 1 * 1, 192, 192,
                    ((64 + 64 + 64 + 96) + 384 + 96) * 128 * 1 * 1, 128, 128,
                    128 * 128 * 1 * 7, 128, 128, 128 * 192 * 7 * 1, 192, 192,
                    ((64 + 64 + 64 + 96) + 384 + 96) * 128 * 1 * 1, 128, 128,
                    128 * 128 * 7 * 1, 128, 128, 128 * 128 * 1 * 7, 128, 128,
                    128 * 128 * 7 * 1, 128, 128, 128 * 192 * 1 * 7, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 160 * 1 * 1, 160, 160,
                    160 * 160 * 1 * 7, 160, 160, 160 * 192 * 7 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 160 * 1 * 1, 160, 160,
                    160 * 160 * 7 * 1, 160, 160, 160 * 160 * 1 * 7, 160, 160,
                    160 * 160 * 7 * 1, 160, 160, 160 * 192 * 1 * 7, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 160 * 1 * 1, 160, 160,
                    160 * 160 * 1 * 7, 160, 160, 160 * 192 * 7 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 160 * 1 * 1, 160, 160,
                    160 * 160 * 7 * 1, 160, 160, 160 * 160 * 1 * 7, 160, 160,
                    160 * 160 * 7 * 1, 160, 160, 160 * 192 * 1 * 7, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    192 * 192 * 1 * 7, 192, 192, 192 * 192 * 7 * 1, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    192 * 192 * 7 * 1, 192, 192, 192 * 192 * 1 * 7, 192, 192,
                    192 * 192 * 7 * 1, 192, 192, 192 * 192 * 1 * 7, 192, 192,
                    (192 + 192 + 192 + 192) * 128 * 1 * 1, 128, 128,
                    128 * 768 * 5 * 5, 768, 768, 768 * 1001, 1001,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    192 * 192 * 1 * 7, 192, 192, 192 * 192 * 7 * 1,
                    192, 192, 192 * 192 * 3 * 3, 192, 192,
                    (192 + 192 + 192 + 192) * 192 * 1 * 1, 192, 192,
                    192 * 320 * 3 * 3, 320, 320,
                    (4 * 192 + 192 + 320) * 320 * 1 * 1, 320, 320,
                    (4 * 192 + 192 + 320) * 384 * 1 * 1, 384, 384,
                    384 * 384 * 1 * 3, 384, 384, 384 * 384 * 3 * 1, 384, 384,
                    (4 * 192 + 192 + 320) * 448 * 1 * 1, 448, 448,
                    448 * 384 * 3 * 3, 384, 384,
                    384 * 384 * 1 * 3, 384, 384, 384 * 384 * 3 * 1, 384, 384,
                    (4 * 192 + 192 + 320) * 192, 192, 192,
                    (320 + 384 * 2 + 384 * 2 + 192) * 320 * 1 * 1, 320, 320,
                    (320 + 384 * 2 + 384 * 2 + 192) * 384 * 1 * 1, 384, 384,
                    384 * 384 * 1 * 3, 384, 384, 384 * 384 * 3 * 1, 384, 384,
                    (320 + 384 * 2 + 384 * 2 + 192) * 448 * 1 * 1, 448, 448,
                    448 * 384 * 3 * 3, 384, 384, 384 * 384 * 1 * 3, 384,
                    384, 384 * 384 * 3 * 1, 384, 384,
                    (320 + 384 * 2 + 384 * 2 + 192) * 192, 192, 192,
                    2048 * 1001, 1001
                ])
            for init_op in [
                    self.imagenet_inception_v3.train_init_op,
                    self.imagenet_inception_v3.test_init_op,
                    self.imagenet_inception_v3.train_eval_init_op
            ]:
                sess.run(init_op)
                losses_, regularizer_, accuracy_ = sess.run([
                    self.imagenet_inception_v3.losses,
                    self.imagenet_inception_v3.regularizer,
                    self.imagenet_inception_v3.accuracy
                ])
                self.assertEqual(losses_.shape, (self.batch_size, ))
                self.assertIsInstance(regularizer_, np.float32)
                self.assertIsInstance(accuracy_, np.float32)


if __name__ == "__main__":
    unittest.main()
