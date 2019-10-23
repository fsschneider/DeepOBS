# -*- coding: utf-8 -*-
"""Tests for the All-CNN-C architecture on the CIFAR-100 dataset."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems


class Cifar100_AllCNNCTest(unittest.TestCase):
    """Test for the All-CNN-C architecture on the CIFAR-100 dataset."""

    def setUp(self):
        """Sets up CIFAR-100 dataset for the tests."""
        self.batch_size = 100
        self.cifar100_allcnnc = testproblems.cifar100_allcnnc(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.cifar100_allcnnc.set_up()

        num_param = []
        for parameter in self.cifar100_allcnnc.net.parameters():
            num_param.append(parameter.numel())

        expected_num_param = [
                3 * 96 * 3 * 3, 96, 96 * 96 * 3 * 3, 96, 96 * 96 * 3 * 3, 96,
                96 * 192 * 3 * 3, 192, 192 * 192 * 3 * 3, 192,
                192 * 192 * 3 * 3, 192, 192 * 192 * 3 * 3, 192,
                192 * 192 * 1 * 1, 192, 192 * 100 * 1 * 1, 100
            ]

        self.assertEqual(num_param, expected_num_param)


if __name__ == "__main__":
    unittest.main()
