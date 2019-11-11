# -*- coding: utf-8 -*-
"""Tests for the VAE on the Fashion-MNIST dataset."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems


class FMNIST_VAETest(unittest.TestCase):
    """Test for the VAE on the Fashion-MNIST dataset."""

    def setUp(self):
        """Sets up Fashion-MNIST dataset for the tests."""
        self.batch_size = 100
        self.fmnist_vae = testproblems.fmnist_vae(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""

        torch.manual_seed(42)
        self.fmnist_vae.set_up()

        num_param = []
        for parameter in self.fmnist_vae.net.parameters():
            num_param.append(parameter.numel())

            # Check if number of parameters per "layer" is equal to what we expect
            # We will write them in the following form:
            # - Conv layer: [input_filter*output_filter*kernel[0]*kernel[1]]
            # - Batch norm: [input, input] (for beta and gamma)
            # - Fully connected: [input*output]
            # - Bias: [dim]

        expected_num_param = [
                1 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64,
                7 * 7 * 64 * 8, 8, 7 * 7 * 64 * 8, 8, 8 * 24, 24, 24 * 49, 49,
                1 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64,
                14 * 14 * 64 * 28 * 28, 28 * 28
            ]

        self.assertEqual(num_param, expected_num_param)

if __name__ == "__main__":
    unittest.main()
