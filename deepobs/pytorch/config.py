# -*- coding: utf-8 -*-
import torch

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0
IS_DETERMINISTIC = True


def get_is_deterministic():
    return IS_DETERMINISTIC


def set_is_deterministic(is_deterministic):
    """Sets whether PyTorch should try to run deterministic.

    Args:
        is_deterministic (bool): If ``True``, this flag sets: \
    ``torch.backends.cudnn.deterministic = True`` \
    ``torch.backends.cudnn.benchmark = False``. \
    However, full determinism is not guaranteed. For more information, see: \
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    global IS_DETERMINISTIC
    IS_DETERMINISTIC = is_deterministic


def get_num_workers():
    return NUM_WORKERS


def set_num_workers(num_workers):
    """Sets the number of workers that are used in the torch DataLoaders.

    Args:
        num_workers (int): The number of workers that are used for data loading.
        """
    global NUM_WORKERS
    NUM_WORKERS = num_workers


def get_default_device():
    return DEFAULT_DEVICE


def set_default_device(device):
    """Sets the device on which the PyTorch experiments are run.

    Args:
        device (str): Device on which to run the PyTorch test problems. E.g. 'cuda' or 'cuda:0'
        """
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device

