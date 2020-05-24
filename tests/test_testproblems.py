"""Tests for all test problems in DeepOBS

Raises:
    ValueError: If we want to test an unknown framework
        (besides PyTorch and TensorFlow)
"""
import os

import numpy as np
import pytest
import tensorflow as tf
import torch

import deepobs.pytorch.testproblems as torch_testproblems
import deepobs.tensorflow.testproblems as tf_testproblems
from deepobs.pytorch.config import get_default_device, set_default_device

from .utils.utils_tests import (
    check_lists,
    get_number_of_parameters,
    get_testproblems,
)

# Basic Settings of the Test
BATCH_SIZE = 8
NR_PT_TESTPROBLEMS = 20
NR_TF_TESTPROBLEMS = 27
DEVICES = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
FRAMEWORKS = ["pytorch", "tensorflow"]
TEST_PROBLEMS = get_testproblems()

# Collect Testproblems
ALL_CONFIGURATIONS = []
for framework in FRAMEWORKS:
    for prob in TEST_PROBLEMS[framework]:
        ALL_CONFIGURATIONS.append((prob, framework))


def test_number_testproblems():
    """Check the number of implemented testproblems per framework.

    We currently have all 27 testproblems in TensorFlow, but only 20 in PyTorch.
    The test should flag if this number changes.
    """
    assert len(TEST_PROBLEMS["pytorch"]) == NR_PT_TESTPROBLEMS
    assert len(TEST_PROBLEMS["tensorflow"]) == NR_TF_TESTPROBLEMS


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("problem,framework", ALL_CONFIGURATIONS)
def test_testproblems(problem, framework, device):
    """Parametrized test for the test problems of DeepOBS.

    We currently check the forward pass (whether it works, results in correct
    shape and type of losses, loss and accuracy) and the number of parameters in
    the network (split by layer, therefore it checks the shape of the network).
    
    Args:
        problem (str): (Name of) a DeepOBS test problem
        framework (str): A framework supported by DeepOBS
            (pytorch or tensorflow)
        device (str): A device to run the computations on
            (either cpu or cuda for GPU)
    
    Raises:
        ValueError: If we want to test an unknown framework
            (besides PyTorch and TensorFlow)
    """
    if framework == "pytorch":
        set_default_device(device)
        tproblem = getattr(torch_testproblems, problem)(batch_size=BATCH_SIZE)
    elif framework == "tensorflow":
        tf.reset_default_graph()
        tproblem = getattr(tf_testproblems, problem)(batch_size=BATCH_SIZE)
    else:
        raise ValueError("Unknown framework in test.")

    # Check forward pass
    _check_forward_pass(tproblem, framework, device)
    # Check number of parameters
    _check_parameters(tproblem, framework)


def _check_forward_pass(tproblem, framework, device):
    """Checks a forward pass of the testproblem."""
    tproblem.set_up()
    if framework == "pytorch":
        for init_op in [
            tproblem.train_init_op,
            tproblem.train_eval_init_op,
            tproblem.valid_init_op,
            tproblem.test_init_op,
        ]:
            init_op()
            losses, acc = tproblem.get_batch_loss_and_accuracy(reduction="none")
            loss = torch.mean(losses).item()
            check_losses_acc(losses, loss, acc)
    elif framework == "tensorflow":
        if device == "cpu":
            config = tf.ConfigProto(device_count={"GPU": 0})
        else:
            config = tf.ConfigProto(device_count={"GPU": 1})
            config.gpu_options.allow_growth = True
        tf_loss = tf.reduce_mean(tproblem.losses) + tproblem.regularizer
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for init_op in [
                tproblem.train_init_op,
                tproblem.train_eval_init_op,
                tproblem.valid_init_op,
                tproblem.test_init_op,
            ]:
                sess.run(init_op)
                acc = 0  # init to integer 0, overwrite if possible.
                if tproblem.accuracy is None:
                    losses, loss, regularizer = sess.run(
                        [tproblem.losses, tf_loss, tproblem.regularizer]
                    )
                else:
                    losses, loss, regularizer, acc = sess.run(
                        [
                            tproblem.losses,
                            tf_loss,
                            tproblem.regularizer,
                            tproblem.accuracy,
                        ]
                    )
                assert isinstance(regularizer, np.float32)
                _check_losses_acc(losses, loss, acc)
    else:
        raise ValueError("Unknown framework in test.")


def _check_losses_acc(losses, loss, acc):
    """Checks whether the losses, the loss and the accuracy have the right shape."""
    assert len(losses) == BATCH_SIZE
    assert isinstance(loss, float) or isinstance(loss, np.float32)
    # check that accuracy is float. If there is no accuracy,
    # the type is an integer and we always set it to zero.
    assert (
        isinstance(acc, float)
        or isinstance(loss, np.float32)
        or (isinstance(acc, int) and acc == 0)
    )
    assert acc >= 0.0
    assert acc <= 1.0


def _check_parameters(tproblem, framework):
    """Checks whether the number of parameters of the network is correct."""
    num_param = []

    if framework == "pytorch":
        for parameter in tproblem.net.parameters():
            num_param.append(parameter.numel())
    elif framework == "tensorflow":
        num_param = [
            np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()
        ]

    assert check_lists(num_param, get_number_of_parameters(tproblem))
