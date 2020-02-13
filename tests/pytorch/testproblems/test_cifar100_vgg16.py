# -*- coding: utf-8 -*-
"""Tests for the VGG16 architecture on the CIFAR-100 dataset."""

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


class Cifar100_VGG16Test(unittest.TestCase):
    """Test for the VGG16 architecture on the CIFAR-100 dataset."""

    def setUp(self):
        """Sets up CIFAR-100 dataset for the tests."""
        self.batch_size = 100
        self.cifar100_vgg16 = testproblems.cifar100_vgg16(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.cifar100_vgg16.set_up()

        num_param = []
        for parameter in self.cifar100_vgg16.net.parameters():
            num_param.append(parameter.numel())

        expected_num_param = [
            3 * 64 * 3 * 3,
            64,
            64 * 64 * 3 * 3,
            64,
            64 * 128 * 3 * 3,
            128,
            128 * 128 * 3 * 3,
            128,
            128 * 256 * 3 * 3,
            256,
            256 * 256 * 3 * 3,
            256,
            256 * 256 * 3 * 3,
            256,
            256 * 512 * 3 * 3,
            512,
            512 * 512 * 3 * 3,
            512,
            512 * 512 * 3 * 3,
            512,
            512 * 512 * 3 * 3,
            512,
            512 * 512 * 3 * 3,
            512,
            512 * 512 * 3 * 3,
            512,
            512 * 7 * 7 * 4096,
            4096,
            4096 * 4096,
            4096,
            4096 * 100,
            100,
        ]

        self.assertEqual(num_param, expected_num_param)


if __name__ == "__main__":
    unittest.main()
