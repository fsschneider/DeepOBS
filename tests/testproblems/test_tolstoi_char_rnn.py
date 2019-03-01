# -*- coding: utf-8 -*-
"""Tests for the Char RNN architecture for the Tolstoi dataset."""

import os
import sys
import unittest
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import testproblems


class Tolstoi_Char_RNNTest(unittest.TestCase):
    """Tests for the Char RNN architecture for the Tolstoi dataset."""

    def setUp(self):
        """Sets up Tolstoi dataset for the tests."""
        self.batch_size = 100
        self.tolstoi_char_rnn = testproblems.tolstoi_char_rnn(self.batch_size)

    def test_init_ops(self):
        """Tests all three initialization operations."""
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.tolstoi_char_rnn.set_up()
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
                83 * 128, 4 * (128 * 128 + 128 * 128), 4 * 128,
                4 * (128 * 128 + 128 * 128), 4 * 128, 83 * 128, 83
            ])
            for init_op in [
                    self.tolstoi_char_rnn.train_init_op,
                    self.tolstoi_char_rnn.test_init_op,
                    self.tolstoi_char_rnn.train_eval_init_op
            ]:
                sess.run(init_op)
                losses_, regularizer_, accuracy_ = sess.run([
                    self.tolstoi_char_rnn.losses,
                    self.tolstoi_char_rnn.regularizer,
                    self.tolstoi_char_rnn.accuracy
                ])
                self.assertEqual(losses_.shape, (self.batch_size, ))
                self.assertIsInstance(regularizer_, np.float32)
                self.assertIsInstance(accuracy_, np.float32)


if __name__ == "__main__":
    unittest.main()
