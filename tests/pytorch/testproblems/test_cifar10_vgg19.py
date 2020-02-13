# -*- coding: utf-8 -*-
"""Tests for the VGG19 architecture on the CIFAR-10 dataset."""

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


class Cifar10_VGG19Test(unittest.TestCase):
    """Test for the VGG19 architecture on the CIFAR-10 dataset."""

    def setUp(self):
        """Sets up CIFAR-10 dataset for the tests."""
        self.batch_size = 100
        self.cifar10_vgg19 = testproblems.cifar10_vgg19(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.cifar10_vgg19.set_up()

        num_param = []
        for parameter in self.cifar10_vgg19.net.parameters():
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
            512 * 512 * 3 * 3,
            512,
            512 * 512 * 3 * 3,
            512,
            512 * 7 * 7 * 4096,
            4096,
            4096 * 4096,
            4096,
            4096 * 10,
            10,
        ]

        self.assertEqual(num_param, expected_num_param)


if __name__ == "__main__":
    unittest.main()
