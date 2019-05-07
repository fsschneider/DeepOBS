# -*- coding: utf-8 -*-
"""Tests for the 3c3d architecture on the CIFAR-10 dataset."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems


class Cifar10_3c3dTest(unittest.TestCase):
    """Test for the 3c3d architecture on the CIFAR-10 dataset."""

    def setUp(self):
        """Sets up CIFAR-10 dataset for the tests."""
        self.batch_size = 100
        self.cifar10_3c3d = testproblems.cifar10_3c3d(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.cifar10_3c3d.set_up()

        num_param = []
        for parameter in self.cifar10_3c3d.net.parameters():
            num_param.append(parameter.numel())

        expected_num_param = [
                3 * 64 * 5 * 5, 64, 64 * 96 * 3 * 3, 96, 96 * 128 * 3 * 3, 128,
                3 * 3 * 128 * 512, 512, 512 * 256, 256, 256 * 10, 10
            ]

        self.assertEqual(num_param, expected_num_param)




if __name__ == "__main__":
    unittest.main()
