# -*- coding: utf-8 -*-
"""Tests for the Char RNN architecture for the Tolstoi dataset."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems


class Tolstoi_Char_RNNTest(unittest.TestCase):
    """Tests for the Char RNN architecture for the Tolstoi dataset."""

    def setUp(self):
        """Sets up Tolstoi dataset for the tests."""
        self.batch_size = 100
        self.tolstoi_char_rnn = testproblems.tolstoi_char_rnn(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.tolstoi_char_rnn.set_up()

        num_param = []
        for parameter in self.tolstoi_char_rnn.net.parameters():
            num_param.append(parameter.numel())

        # form of parameters:
        # bias hidden-hidden calculation: 4*hidden_size
        # weight hidden-hidden: [4*hidden, hidden]
        # bias input-hidden: 4*input_size
        # weight input-hidden: [4*input, hidden]
        expected_num_param = [
                83 * 128, 4 * 128 * 128, 4 * 128 * 128, 4 * 128, 4* 128,
                4 * 128 * 128, 4 * 128 * 128, 4 * 128, 4 * 128, 83 * 128, 83
            ]

        self.assertEqual(num_param, expected_num_param)


if __name__ == "__main__":
    unittest.main()
