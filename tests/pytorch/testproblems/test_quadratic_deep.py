# -*- coding: utf-8 -*-
"""Tests for the deep quadratic loss function."""

import os
import sys
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.pytorch import testproblems


class Quadratic_DeepTest(unittest.TestCase):
    """Tests for the deep quadratic loss function."""

    def setUp(self):
        """Sets up the quadratic dataset for the tests."""
        self.batch_size = 10
        self.quadratic_deep = testproblems.quadratic_deep(self.batch_size)

    def test_num_param(self):
        """Tests the number of parameters."""
        torch.manual_seed(42)
        self.quadratic_deep.set_up()
        for parameter in self.quadratic_deep.net.parameters():
            self.assertEqual(parameter.numel(), 100)
    def test_hessian_sqrt(self):
        hessian = testproblems.quadratic_deep._make_hessian()
        sqrt = testproblems.testproblems_modules.net_quadratic_deep._compute_sqrt(
            hessian)
        check_hessian = torch.einsum("ij,kj->ik", (sqrt, sqrt))
        assert torch.allclose(hessian, check_hessian, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
