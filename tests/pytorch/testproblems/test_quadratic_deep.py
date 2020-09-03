# -*- coding: utf-8 -*-
"""Tests for the deep quadratic loss function."""

import os
import sys
import unittest

import numpy as np
import torch

from deepobs.pytorch import testproblems

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)


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
            if parameter.requires_grad:
                self.assertEqual(parameter.numel(), 100)

    def test_forward(self):
        quadratic_deep = testproblems.quadratic_deep(self.batch_size)
        # prob = self.quadratic_deep
        prob = quadratic_deep

        prob.set_up()
        prob.train_init_op()

        inputs, labels = prob._get_next_batch()
        num_inputs = inputs.numel()
        net, loss_function = prob.net, prob.loss_function

        # from test problem
        outputs = net(inputs)
        loss = loss_function(reduction="mean")(outputs, labels)

        # manual re-computation of the model output
        shifted = -inputs + net.shift.bias.data
        sqrt = testproblems.testproblems_modules.net_quadratic_deep._compute_sqrt(
            prob._hessian)
        outputs_check = torch.einsum('ji,bj->bi', (sqrt, shifted))
        assert torch.allclose(outputs, outputs_check, atol=1e-6, rtol=1e-6)

        # check Hessian initialization
        hessian_check = torch.einsum(
            "ki,kj->ij", (net.scale.weight.data, net.scale.weight.data))
        assert torch.allclose(prob._hessian,
                              hessian_check,
                              atol=1e-6,
                              rtol=1e-6)

        # manual recomputation of the loss
        loss_check = torch.einsum(
            "bi,ij,bj", (shifted, prob._hessian, shifted)) / num_inputs
        assert torch.allclose(loss, loss_check)

    def test_hessian_sqrt(self):
        hessian = testproblems.quadratic_deep._make_hessian()
        sqrt = testproblems.testproblems_modules.net_quadratic_deep._compute_sqrt(
            hessian)
        check_hessian = torch.einsum("ij,kj->ik", (sqrt, sqrt))
        assert torch.allclose(hessian, check_hessian, rtol=1e-6, atol=1e-6)

    def test_hessian_deterministic(self):
        hessian1 = self.quadratic_deep._make_hessian()
        hessian2 = self.quadratic_deep._make_hessian()
        assert torch.allclose(hessian1, hessian2)


if __name__ == "__main__":
    unittest.main()
