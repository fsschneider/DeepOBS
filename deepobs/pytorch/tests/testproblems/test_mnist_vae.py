# -*- coding: utf-8 -*-
"""Tests for the VAE on the MNIST dataset."""

import os
import sys
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems


class MNIST_VAETest(unittest.TestCase):
    """Test for the VAE on the MNIST dataset."""

    def setUp(self):
        """Sets up MNIST dataset for the tests."""
        self.batch_size = 100
        self.mnist_vae = testproblems.mnist_vae(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.mnist_vae.set_up()

        num_param = []
        for parameter in self.mnist_vae.net.parameters():
            num_param.append(parameter.numel())

        expected_num_param = [
                1 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64,
                7 * 7 * 64 * 8, 8, 7 * 7 * 64 * 8, 8, 8 * 24, 24, 24 * 49, 49,
                1 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64, 64 * 64 * 4 * 4, 64,
                14 * 14 * 64 * 28 * 28, 28 * 28
        ]

        self.assertEqual(num_param, expected_num_param)

if __name__ == "__main__":
    unittest.main()
