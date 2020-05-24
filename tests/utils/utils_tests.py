"""General utility functions for tests"""

import deepobs.pytorch.datasets as torch_datasets
import deepobs.pytorch.testproblems as torch_testproblems
import deepobs.tensorflow.datasets as tf_datasets
import deepobs.tensorflow.testproblems as tf_testproblems

from .utils_number_of_parameters import NUMBER_OF_PARAMETERS


def check_lists(list1, list2):
    """Checks whether two lists have equal elements
    
    Args:
        list1 (list): First list
        list2 (list): Second list
    
    Returns:
        bool: Bool whether they have equal elements
    """
    return all([a == b for a, b in zip(list1, list2)])


def get_testproblems():
    """Get lists of all available test problems per framework
    
    Returns:
        dict: A dict where the key is a framework and the value is a list of
            available test problems
    """
    torch_probs = dir(torch_testproblems)
    tf_probs = dir(tf_testproblems)

    # Clean up some utils function from testproblems
    tf_probs = [
        tp
        for tp in tf_probs
        if not tp.startswith("_") and not tp.lower().startswith("test")
    ]
    torch_probs = [
        tp
        for tp in torch_probs
        if not tp.startswith("_") and not tp.lower().startswith("test")
    ]
    return {"pytorch": torch_probs, "tensorflow": tf_probs}


def get_datasets():
    """Get lists of all available data sets per framework
    
    Returns:
        dict: A dict where the key is a framework and the value is a list of
            available data sets
    """
    torch_probs = dir(torch_datasets)
    tf_probs = dir(tf_datasets)

    # Clean up some utils function from testproblems
    tf_probs = [
        tp
        for tp in tf_probs
        if not tp.startswith("_") and not tp.lower().startswith("dataset")
    ]
    torch_probs = [
        tp
        for tp in torch_probs
        if not tp.startswith("_") and not tp.lower().startswith("dataset")
    ]
    return {"pytorch": torch_probs, "tensorflow": tf_probs}


def get_number_of_parameters(problem):
    """Returns a list of parameters to expect for a test problem
    
    Args:
        problem (str): (Name of) a DeepOBS test problem
    
    Returns:
        list: Number of parameters of the network per layer
    """
    return NUMBER_OF_PARAMETERS[problem.__class__.__name__]
