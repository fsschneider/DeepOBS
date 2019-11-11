# -*- coding: utf-8 -*-
FRAMEWORK = "pytorch"
BASELINE_DIR = "baselines_deepobs"
SMALL_TEST_SET = ["quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"]
LARGE_TEST_SET = [
    "fmnist_vae",
    "cifar100_allcnnc",
    "svhn_wrn_164",
    "tolstoi_char_rnn",
]
DATA_SET_NAMING = {
    "two": "2D",
    "quadratic": "Quadratic",
    "mnist": "MNIST",
    "fmnist": "F-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "svhn": "SVHN",
    "imagenet": "ImageNet",
    "tolstoi": "Tolstoi",
}
TP_NAMING = {
    "d_beale": "Beale",
    "d_branin": "Branin",
    "d_rosenbrock": "Rosenbrock",
    "deep": "Deep",
    "logreg": "Log. Reg.",
    "mlp": "MLP",
    "2c2d": "2c2d",
    "3c3d": "3c3d",
    "vae": "VAE",
    "vgg_16": "VGG 16",
    "vgg_19": "VGG 19",
    "allcnnc": "All-CNN-C",
    "wrn164": "Wide ResNet 16-4",
    "wrn404": "Wide ResNet 40-4",
    "inception_v3": "Inception-v3",
    "char_rnn": "Char RNN",
}
DATA_DIR = "data_deepobs"


def get_data_dir():
    return DATA_DIR


def set_data_dir(data_dir):
    """Sets the data directory.

    Args:
        data_dir (str): Path to the data folder.
    """
    global DATA_DIR
    DATA_DIR = data_dir


def get_framework():
    return FRAMEWORK


def set_framework(framework):
    """Sets the current used framework. This is relevant for the higher level Tuner module of DeepOBS.

    Args:
        framework (str): Can be 'pytorch' or 'tensorflow'
    """
    global FRAMEWORK
    FRAMEWORK = framework


def get_baseline_dir():
    return BASELINE_DIR


def set_baseline_dir(baseline_dir):
    global BASELINE_DIR
    BASELINE_DIR = baseline_dir


def get_small_test_set():
    return SMALL_TEST_SET


def set_small_test_set(testset):
    global SMALL_TEST_SET
    SMALL_TEST_SET = testset


def get_large_test_set():
    return LARGE_TEST_SET


def set_large_test_set(testset):
    global LARGE_TEST_SET
    LARGE_TEST_SET = testset


def get_data_set_naming():
    return DATA_SET_NAMING


def get_tp_naming():
    return TP_NAMING


DEFAULT_TEST_PROBLEMS_SETTINGS = {
    "quadratic_deep": {"batch_size": 128, "num_epochs": 100},
    "mnist_vae": {"batch_size": 64, "num_epochs": 50},
    "fmnist_2c2d": {"batch_size": 128, "num_epochs": 100},
    "cifar10_3c3d": {"batch_size": 128, "num_epochs": 100},
    "fmnist_vae": {"batch_size": 64, "num_epochs": 100},
    "cifar100_allcnnc": {"batch_size": 256, "num_epochs": 350},
    "svhn_wrn164": {"batch_size": 128, "num_epochs": 160},
    "tolstoi_char_rnn": {"batch_size": 50, "num_epochs": 200},
    "mnist_2c2d": {"batch_size": 128, "num_epochs": 100},
    "mnist_mlp": {"batch_size": 128, "num_epochs": 100},
    "fmnist_mlp": {"batch_size": 128, "num_epochs": 100},
    "mnist_logreg": {"batch_size": 128, "num_epochs": 50},
}


def get_testproblem_default_setting(testproblem):
    """Returns default settings for the ``batch_size`` and the ``num_epochs`` for ``testproblem`` (if available).

    Args:
        testproblem (str): Test problem for which to return the default setting.

    Returns:
        dict: A dictionary with the default values for ``batch_size`` and ``num_epochs``
        """

    try:
        return DEFAULT_TEST_PROBLEMS_SETTINGS[testproblem]
    except KeyError:
        raise RuntimeError(
            "There are no default settings for batch_size and num_epochs for testproblem "
            + testproblem
            + ". Please set num_epochs and batch_size in the run."
        )

