"""Utility functions for testing DeepOBS models."""


def _check_lists(list1, list2):
    """Checks whether two lists have equal elements.

    Args:
        list1 (list): First list
        list2 (list): Second list

    Returns:
        bool: Bool whether they have equal elements
    """
    return all([a == b for a, b in zip(list1, list2)])
