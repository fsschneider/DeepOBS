# -*- coding: utf-8 -*-
"""Base class for DeepOBS test problems."""
import torch
import abc
from .. import config


class TestProblem(abc.ABC):
    """Base class for DeepOBS test problems.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): Weight decay (L2-regularization) factor to use. If
        not specified, the test problems revert to their respective defaults.
        Note: Some test problems do not use regularization and this value will
        be ignored in such a case.

  Attributes:
    _batch_size: Batch_size for the data of this test problem.
    _weight_decay: The regularization factor for this test problem
    data: The dataset used by the test problem (datasets.DataSet instance).
    loss_function: The loss function for this test problem.
    net: The torch module (the neural network) that is trained.

  Methods:
    train_init_op: Initializes the test problem for the
        training phase.
    train_eval_init_op: Initializes the test problem for
        evaluating on training data.
    test_init_op: Initializes the test problem for
        evaluating on test data.
    _get_next_batch: Returns the next batch of data of the current phase.
    get_batch_loss_and_accuracy: Calculates the loss and accuracy of net on the next batch of the current phase.
    set_up: Sets all public attributes.
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

        self._device = config.get_default_device()

        # Public attributes by which to interact with test problems. These have to
        # be created by the set_up function of sub-classes.
        self.data = None
        self.loss_function = None
        self.net = None

    def train_init_op(self):
        """Initializes the testproblem instance to train mode. I.e.
        sets the iterator to the training set and sets the model to train mode.
        """
        self._iterator = iter(self.data._train_dataloader)
        self.phase = "train"
        self.net.train()

    def train_eval_init_op(self):
        """Initializes the testproblem instance to train eval mode. I.e.
        sets the iterator to the train evaluation set and sets the model to eval mode.
        """
        self._iterator = iter(self.data._train_eval_dataloader)
        self.phase = "train_eval"
        self.net.eval()

    def test_init_op(self):
        """Initializes the testproblem instance to test mode. I.e.
        sets the iterator to the test set and sets the model to eval mode.
        """
        self._iterator = iter(self.data._test_dataloader)
        self.phase = "test"
        self.net.eval()

    def _get_next_batch(self):
        """Returns the next batch from the iterator."""
        return next(self._iterator)

    def get_batch_loss_and_accuracy(self):
        """Gets a new batch and calculates the loss and accuracy (if available)
        on that batch. This is a default implementation for image classification.
        Testproblems with different calculation routines (e.g. RNNs) overwrite this method accordingly."""

        inputs, labels = self._get_next_batch()
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)
        correct = 0.0
        total = 0.0

        # in evaluation phase is no gradient needed
        if self.phase in ["train_eval", "test"]:
            with torch.no_grad():
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
        else:
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = correct/total
        return loss, accuracy

    @abc.abstractmethod
    def set_up(self):
        """Sets up the test problem.
        """
        pass