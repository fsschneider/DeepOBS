"""Configuration file for DeepOBS."""

DATA_DIR = "data_deepobs"

DEVICE = "cuda"
DETERMINISTIC_COMPUTATION = True
NUM_DATA_WORKERS = 0


def get_data_dir():
    """Get the current data directory.

    Returns:
        str: Path to the data folder
    """
    return DATA_DIR


def set_data_dir(data_dir):
    """Set the data directory.

    Args:
        data_dir (str): Path to the data folder
    """
    global DATA_DIR
    DATA_DIR = data_dir


def get_default_device():
    """Get the default device for the computation, which by default is "cuda".

    Returns:
        str: Device on which the Problems are run.
    """
    return DEVICE


def set_default_device(device):
    """Sets the device on which the experiments are run.

    Args:
        device (str): Device on which to run the problems. E.g. 'cuda' or 'cpu'.
    """
    global DEVICE
    DEVICE = device


def get_deterministic_computation():
    """Check whether computation should be deterministic, which it is by default.

    Returns:
        bool: Whether computation should be performed (almost) deterministicly.
    """
    return DETERMINISTIC_COMPUTATION


def set_deterministic_computation(compute_deterministic):
    """Set whether computation should be deterministic.

    Args:
        compute_deterministic (bool): If ``True``, then try to compute deterministicly.
            This involves setting flags such as ``torch.backends.cudnn.deterministic``,
            depending on the framework.
    """
    global DETERMINISTIC_COMPUTATION
    DETERMINISTIC_COMPUTATION = compute_deterministic


def get_num_workers():
    """Get the number of workers used for the Data Loaders.

    Returns:
        int: The number of workers that are used for data loading.
    """
    return NUM_DATA_WORKERS


def set_num_workers(num_workers):
    """Sets the number of workers that are used for the Data Loaders.

    Args:
        num_workers (int): The number of workers that are used for data loading.
    """
    global NUM_DATA_WORKERS
    NUM_DATA_WORKERS = num_workers
