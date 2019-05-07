# -*- coding: utf-8 -*-
"""Base class for DeepOBS datasets."""

# pylint: disable=too-many-instance-attributes, too-few-public-methods
class DataSet(object):
    """Base class for DeepOBS data sets.

  Args:
    batch_size (int): The mini-batch size to use.

  Methods:
     _make_train_dataloader: Creates a torch data loader for the training data with batches of size batch_size.
     _make_train_eval_dataloader: Creates a torch data loader for the training evaluation data with batches of size batch_size.
     _make_test_dataloader: Creates a torch data loader for the test data with batches of size batch_size.

  """

    def __init__(self, batch_size):
        """Creates a new DataSet instance.

    Args:
      batch_size (int): The mini-batch size to use.
    """
        self._batch_size = batch_size
        self._train_dataloader = self._make_train_dataloader()
        self._train_eval_dataloader = self._make_train_eval_dataloader()
        self._test_dataloader = self._make_test_dataloader()

    def _make_train_dataloader(self):
        """Creates the training data loader.

    Returns:
      A torch.utils.data.DataLoader instance with batches of training data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")

    def _make_train_eval_dataloader(self):
        """Creates the train eval data loader.

    Returns:
      A torch.utils.data.DataLoader instance with batches of training eval data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")

    def _make_test_dataloader(self):
        """Creates the test data loader.

    Returns:
      A torch.utils.data.DataLoader instance with batches of test data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")
