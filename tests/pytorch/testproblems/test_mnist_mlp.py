# -*- coding: utf-8 -*-
"""Tests for the MLP on the MNIST dataset."""

import os
import sys
import unittest

import torch

from deepobs.pytorch import testproblems

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)



class MNIST_MLPTest(unittest.TestCase):
    """Test for the MLP on the MNIST dataset."""

    def setUp(self):
        """Sets up Fashion-MNIST dataset for the tests."""
        self.batch_size = 100
        self.mnist_mlp = testproblems.mnist_mlp(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""

        torch.manual_seed(42)
        self.mnist_mlp.set_up()

        num_param = []
        for parameter in self.mnist_mlp.net.parameters():
            num_param.append(parameter.numel())

            # Check if number of parameters per "layer" is equal to what we expect
            # We will write them in the following form:
            # - Conv layer: [input_filter*output_filter*kernel[0]*kernel[1]]
            # - Batch norm: [input, input] (for beta and gamma)
            # - Fully connected: [input*output]
            # - Bias: [dim]

        expected_num_param = [
            28 * 28 * 1000,
            1000,
            1000 * 500,
            500,
            500 * 100,
            100,
            100 * 10,
            10,
        ]

        self.assertEqual(num_param, expected_num_param)


if __name__ == "__main__":
    unittest.main()
