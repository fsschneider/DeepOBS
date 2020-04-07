# -*- coding: utf-8 -*-
"""Tests for the Wide ResNet 40-4 architecture on the CIFAR-100 dataset."""

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


class Cifar100_WRN404Test(unittest.TestCase):
    """Test for the Wide ResNet 40-4 architecture on the CIFAR-100 dataset."""

    def setUp(self):
        """Sets up CIFAR-100 dataset for the tests."""
        self.batch_size = 100
        self.cifar100_wrn404 = testproblems.cifar100_wrn404(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.cifar100_wrn404.set_up()

        num_param = []
        for parameter in self.cifar100_wrn404.net.parameters():
            num_param.append(parameter.numel())

        expected_num_param = [
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
            64 * 64 * 3 * 3,
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
            64 * 64 * 3 * 3,
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
            128 * 128 * 3 * 3,
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
            128 * 128 * 3 * 3,
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
            256 * 256 * 3 * 3,
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
            256 * 256 * 3 * 3,
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
            256 * 100,
            100,
        ]

        self.assertEqual(num_param, expected_num_param)


if __name__ == "__main__":
    unittest.main()
