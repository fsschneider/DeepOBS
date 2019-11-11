# -*- coding: utf-8 -*-
"""Tests for the 2c2d architecture on the Fashion-MNIST dataset."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems


class FMNIST_2c2dTest(unittest.TestCase):
    """Test for the 2c2d architecture on the Fashion-MNIST dataset."""

    def setUp(self):
        """Sets up Fashion-MNIST dataset for the tests."""
        self.batch_size = 100
        self.fmnist_2c2d = testproblems.fmnist_2c2d(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters"""
        torch.manual_seed(42)
        self.fmnist_2c2d.set_up()

        num_param = []
        for parameter in self.fmnist_2c2d.net.parameters():
            num_param.append(parameter.numel())

            # Check if number of parameters per "layer" is equal to what we expect
            # We will write them in the following form:
            # - Conv layer: [input_filter*output_filter*kernel[0]*kernel[1]]
            # - Batch norm: [input, input] (for beta and gamma)
            # - Fully connected: [input*output]
            # - Bias: [dim]

        expected_num_param = [
                1 * 32 * 5 * 5, 32, 32 * 64 * 5 * 5, 64, 7 * 7 * 64 * 1024,
                1024, 1024 * 10, 10
            ]

        self.assertEqual(num_param, expected_num_param)


if __name__ == "__main__":
    unittest.main()
