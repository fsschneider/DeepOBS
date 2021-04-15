"""Tests for DeepOBS models."""

import inspect

import numpy as np
import pytest

import deepobs.models.pytorch as pytorch_models
import deepobs.models.tensorflow as tensorflow_models
from deepobs.models import info
from tests.test_models import _utils

# DEVICES = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
FRAMEWORKS = [
    "pytorch",
    # "tensorflow",
]

# Verification values
NR_PT_MODELS = 11
NR_TF_MODELS = 0

# Collect all possible test scenarios (devices, frameworks, etc.)
SCENARIOS = []
SCENARIO_IDS = []
for fw in FRAMEWORKS:
    models = globals()[fw + "_models"].__all__
    SCENARIOS.extend([(fw, model) for model in models])
    SCENARIO_IDS.extend([fw + ":" + ds for ds in models])


def test_number_datasets_per_framework():
    """Check the number of implemented data sets per framework.

    We currently have all 2 data sets in PyTorch, and 0 in TensorFlow.
    The test should flag if this number changes.
    """
    assert len(pytorch_models.__all__) == NR_PT_MODELS
    assert len(tensorflow_models.__all__) == NR_TF_MODELS


@pytest.mark.parametrize("framework, model", SCENARIOS, ids=SCENARIO_IDS)
def test_model(framework, model):
    """Test the data set.

    Args:
        framework (str): String of the framework to test.
        model (str): String of the model to test.
    """
    # Load DataSet
    net_cls = getattr(globals()[fw + "_models"], model)
    net_args = []
    for arg in inspect.getfullargspec(net_cls)[0]:
        if arg == "self":
            pass
        elif arg == "num_residual_blocks":
            n_blocks = getattr(info, model.upper() + "_DEFAULT_NUM_BLOCKS")
            net_args.append(n_blocks)
        elif arg == "widening_factor":
            widening_factor = getattr(info, model.upper() + "_DEFAULT_WIDENING_FACTOR")
            net_args.append(widening_factor)
        elif arg == "num_outputs":
            n_classes = getattr(info, model.upper() + "_DEFAULT_CLASSES")
            net_args.append(n_classes)
        elif arg == "num_channels":
            num_channels = getattr(info, model.upper() + "_DEFAULT_CHANNELS")
            net_args.append(num_channels)
        elif arg == "variant":
            variant = getattr(info, model.upper() + "_DEFAULT_VARIANT")
            net_args.append(variant)
        elif arg == "n_latent":
            n_latent = getattr(info, model.upper() + "_DEFAULT_LATENT_SPACE")
            net_args.append(n_latent)
        elif arg == "hessian":
            hessian = np.diag(100 * [1])
            net_args.append(hessian)

    net = net_cls(*net_args)
    model_info = getattr(info, model.upper())

    _verify_parameters(net, model_info, framework)


def _verify_parameters(net, model_info, framework):
    """Verifies that the number of parameters of the network is correct.

    Args:
        net (Model): A DeepOBS model.
        model_info (dict): A dictionary holding information about the net.
        framework (str): String of the framework to test.
    """
    num_param = []

    if framework == "pytorch":
        for param in net.parameters():
            if param.requires_grad:
                num_param.append(param.numel())
    # elif framework == "tensorflow":
    # num_param = [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]
    # print(num_param)
    # print(model_info["parameters"])
    assert _utils._check_lists(num_param, model_info["parameters"])
