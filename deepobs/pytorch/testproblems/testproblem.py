# -*- coding: utf-8 -*-
"""Base class for DeepOBS test problems."""

class TestProblem(object):
    """Base class for DeepOBS test problems.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): Weight decay (L2-regularization) factor to use. If
        not specified, the test problems revert to their respective defaults.
        Note: Some test problems do not use regularization and this value will
        be ignored in such a case.

  Attributes:
    dataset: The dataset used by the test problem (datasets.DataSet instance).
    train_init_op: A tensorflow operation initializing the test problem for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the test problem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the test problem for
        evaluating on test data.
    losses: A tf.Tensor of shape (batch_size, ) containing the per-example loss
        values.
    regularizer: A scalar tf.Tensor containing a regularization term (might be
        a constant 0.0 for test problems that do not use regularization).
    accuracy: A scalar tf.Tensor containing the mini-batch mean accuracy.
  """

    def __init__(self, batch_size, weight_decay=None):
        """Creates a new test problem instance.

    Args:
      batch_size (int): Batch size to use.
      weight_decay (float): Weight decay (L2-regularization) factor to use. If
          not specified, the test problems revert to their respective defaults.
          Note: Some test problems do not use regularization and this value will
          be ignored in such a case.
    """
        self._batch_size = batch_size
        self._weight_decay = weight_decay

        # Public attributes by which to interact with test problems. These have to
        # be created by the set_up function of sub-classes.
        self.data = None
        self.loss_function = None
        self.net = None

    def train_init_op(self):
        self._iterator = iter(self.data._train_dataloader)
        self.phase = "train"
        self.net.train()

    def train_eval_init_op(self):
        self._iterator = iter(self.data._train_eval_dataloader)
        self.phase = "train_eval"
        self.net.eval()

    def test_init_op(self):
        self._iterator = iter(self.data._test_dataloader)
        self.phase = "test"
        self.net.eval()

    def _get_next_batch(self):
        return next(self._iterator)

    def get_batch_loss_and_accuracy(self):
        """ Gets a new mini batch from the iterator and returns loss and accuracy on it
        """
        raise NotImplementedError(
            """'TestProblem' is an abstract base class, please
        use one of the sub-classes.""")

    def set_up(self):
        """Sets up the test problem.

    This includes setting up the data loading pipeline for the data set and
    creating the tensorflow computation graph for this test problem
    (e.g. creating the neural network).
    """
        raise NotImplementedError(
            """'TestProblem' is an abstract base class, please
        use one of the sub-classes.""")
