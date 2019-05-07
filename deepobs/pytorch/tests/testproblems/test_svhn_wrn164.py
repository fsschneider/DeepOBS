# -*- coding: utf-8 -*-
"""Tests for the Wide ResNet 16-4 architecture on the SVHN dataset."""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems


class SVHN_WRN164Test(unittest.TestCase):
    """Test for the Wide ResNet 16-4 architecture on the SVHN dataset."""

    def setUp(self):
        """Sets up SVHN dataset for the tests."""
        self.batch_size = 100
        self.svhn_wrn164 = testproblems.svhn_wrn164(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.svhn_wrn164.set_up()

        num_param = []
        for parameter in self.svhn_wrn164.net.parameters():
            num_param.append(parameter.numel())

        expected_num_param = [
                3 * 16 * 3 * 3, 16, 16, 16 * 64 * 1 * 1, 16 * 64 * 3 * 3, 64,
                64, 64 * 64 * 3 * 3, 64, 64, 64 * 64 * 3 * 3, 64, 64,
                64 * 64 * 3 * 3, 64, 64, 64 * 128 * 1 * 1, 64 * 128 * 3 * 3,
                128, 128, 128 * 128 * 3 * 3, 128, 128, 128 * 128 * 3 * 3, 128,
                128, 128 * 128 * 3 * 3, 128, 128, 128 * 256 * 1 * 1,
                128 * 256 * 3 * 3, 256, 256, 256 * 256 * 3 * 3, 256, 256,
                256 * 256 * 3 * 3, 256, 256, 256 * 256 * 3 * 3, 256, 256,
                256 * 10, 10
            ]

        self.assertEqual(num_param, expected_num_param)



if __name__ == "__main__":
    unittest.main()
