"""Tests for DeepOBS data sets."""

import pytest

import deepobs.datasets.pytorch as pytorch_datasets
import deepobs.datasets.tensorflow as tensorflow_datasets
from deepobs.datasets import info

# Basic Settings of the Test
BATCH_SIZE = 8
MODES = ["train", "train_eval", "valid", "test"]
# DEVICES = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
FRAMEWORKS = [
    "pytorch",
    # "tensorflow",
]

# Verification values
NR_PT_DATASETS = 9
NR_TF_DATASETS = 0

# Mark slow datasets that require large downloads
SLOW_DATASETS = ["CelebA", "SVHN", "AFHQ"]

# Collect all possible test scenarios (devices, frameworks, etc.)
SCENARIOS = []
SCENARIO_IDS = []
for fw in FRAMEWORKS:
    datasets = globals()[fw + "_datasets"].__all__
    SCENARIOS.extend(
        [
            (pytest.param(fw, ds, marks=pytest.mark.slow))
            if ds in SLOW_DATASETS
            else (fw, ds)
            for ds in datasets
        ]
    )
    SCENARIO_IDS.extend([fw + ":" + ds for ds in datasets])


def test_number_datasets_per_framework():
    """Check the number of implemented data sets per framework.

    We currently have all 2 data sets in PyTorch, and 0 in TensorFlow.
    The test should flag if this number changes.
    """
    assert len(pytorch_datasets.__all__) == NR_PT_DATASETS
    assert len(tensorflow_datasets.__all__) == NR_TF_DATASETS


@pytest.mark.parametrize("framework, dataset", SCENARIOS, ids=SCENARIO_IDS)
def test_dataset(framework, dataset):
    """Test the data set.

    Args:
        framework (str): String of the framework to test.
        dataset (str): String of the dataset to test.
    """
    # Load DataSet
    data = getattr(globals()[fw + "_datasets"], dataset)(BATCH_SIZE)
    dataset_info = getattr(info, dataset.upper())

    # Run the correct check function
    globals()["_check_" + dataset_info["type"] + "_" + framework](data, dataset_info)


def _check_toy_problem_pytorch(data, dataset_info):
    # check for all four interal "data sets"
    for m in MODES:
        dataloader = getattr(data, "_" + m + "_dataloader")
        iterator = iter(dataloader)
        inputs, labels = next(iterator)

        assert len(inputs.shape) == dataset_info["input_shape"]
        assert inputs.shape[0] == BATCH_SIZE
        if dataset_info["input_shape"] == 2:
            assert inputs.shape[1] == dataset_info["dimension"]
            assert type(inputs[0][0].item()) == dataset_info["input_type"]
        elif dataset_info["input_shape"] == 1:
            assert type(inputs[0].item()) == dataset_info["input_type"]

        assert len(labels.shape) == dataset_info["label_shape"]
        assert labels.shape[0] == BATCH_SIZE
        if dataset_info["label_shape"] == 2:
            assert labels.shape[1] == dataset_info["dimension"]
            assert type(labels[0][0].item()) == dataset_info["label_type"]
        elif dataset_info["label_shape"] == 1:
            assert type(labels[0].item()) == dataset_info["label_type"]


def _check_image_classification_pytorch(data, dataset_info):
    # check for all four interal "data sets"
    for m in MODES:
        dataloader = getattr(data, "_" + m + "_dataloader")
        iterator = iter(dataloader)
        inputs, labels = next(iterator)

        # Check image inputs
        _check_image_inputs_pytorch(inputs, dataset_info)

        # Check label shapes and type
        assert len(labels.shape) == dataset_info["label_shape"]
        assert labels.shape[0] == BATCH_SIZE

        assert type(labels[0].item()) == dataset_info["label_type"]
        assert labels.min() >= 0
        assert labels.max() <= dataset_info["classes"] - 1

        _check_dataloader_size(dataloader, m, dataset_info)


def _check_image_attributes_pytorch(data, dataset_info):
    # check for all four interal "data sets"
    for m in MODES:
        dataloader = getattr(data, "_" + m + "_dataloader")
        iterator = iter(dataloader)
        inputs, labels = next(iterator)

        # Check image inputs
        _check_image_inputs_pytorch(inputs, dataset_info)

        # Check label shapes and type
        assert len(labels.shape) == dataset_info["label_shape"]
        assert labels.shape[0] == BATCH_SIZE

        assert len(labels[0]) == dataset_info["label_length"]
        assert type(labels[0][0].item()) == dataset_info["label_type"]

        # Number of batches in each dataloader:
        assert len(dataloader) == dataset_info["n_" + m] // BATCH_SIZE


def _check_image_inputs_pytorch(inputs, dataset_info):
    # Check input shapes
    # PyTorch uses NCHW by default
    assert len(inputs.shape) == 4
    assert inputs.shape[0] == BATCH_SIZE
    assert inputs.shape[1] == dataset_info["channels"]
    assert inputs.shape[2] == dataset_info["height"]
    assert inputs.shape[3] == dataset_info["width"]
    assert type(inputs[0, 0, 0, 0].item()) == dataset_info["input_type"]


def _check_dataloader_size(dataloader, mode, dataset_info):
    # Number of batches in each dataloader:
    assert len(dataloader) == dataset_info["n_" + mode] // BATCH_SIZE
