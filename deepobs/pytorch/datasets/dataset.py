# -*- coding: utf-8 -*-
"""Base class for DeepOBS datasets."""

# pylint: disable=too-many-instance-attributes, too-few-public-methods
class DataSet(object):
    """Base class for DeepOBS data sets.

  Args:
    batch_size (int): The mini-batch size to use.

  Attributes:
    batch: A tuple of tensors, yielding batches of data from the dataset.
        Executing these tensors raises a ``tf.errors.OutOfRangeError`` after one
        epoch.
    train_init_op: A tensorflow operation initializing the dataset for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the testproblem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the testproblem for
        evaluating on test data.
    phase: A string-value tf.Variable that is set to ``train``, ``train_eval``
        or ``test``, depending on the current phase. This can be used by testproblems
        to adapt their behavior to this phase.
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
        """Creates the training dataset.

    Returns:
      A tf.data.Dataset instance with batches of training data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")

    def _make_train_eval_dataloader(self):
        """Creates the train eval dataset.

    Returns:
      A tf.data.Dataset instance with batches of training eval data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")

    def _make_test_dataloader(self):
        """Creates the test dataset.

    Returns:
      A tf.data.Dataset instance with batches of test data.
    """
        raise NotImplementedError(
            """'DataSet' is an abstract base class, please use
        one of the sub-classes.""")
