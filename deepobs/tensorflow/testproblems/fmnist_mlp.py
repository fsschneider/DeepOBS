# -*- coding: utf-8 -*-
"""A  multi-layer perceptron architecture for Fashion-MNIST."""

import tensorflow as tf

from ._mlp import _mlp
from ..datasets.mnist import mnist
from .testproblem import TestProblem


class fmnist_mlp(TestProblem):
    """DeepOBS test problem class for a multi-layer perceptron neural network\
    on Fashion-MNIST.

  The network is build as follows:

    - Four fully-connected layers with ``1000``, ``500``, ``100`` and ``10``
      units per layer.
    - The first three layers use ReLU activation, and the last one a softmax
      activation.
    - The biases are initialized to ``0.0`` and the weight matrices with
      truncated normal (standard deviation of ``3e-2``)
    - The model uses a cross entropy loss.
    - No regularization is used.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): No weight decay (L2-regularization) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

  Attributes:
    dataset: The DeepOBS data set class for Fashion-MNIST.
    train_init_op: A tensorflow operation initializing the test problem for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the test problem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the test problem for
        evaluating on test data.
    losses: A tf.Tensor of shape (batch_size, ) containing the per-example loss
        values.
    regularizer: A scalar tf.Tensor containing a regularization term.
        Will always be ``0.0`` since no regularizer is used.
    accuracy: A scalar tf.Tensor containing the mini-batch mean accuracy.
  """

    def __init__(self, batch_size, weight_decay=None):
        """Create a new multi-layer perceptron test problem instance on \
        Fashion-MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(fmnist_mlp, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )

    def set_up(self):
        """Set up the multi-layer perceptron test problem instance on
        Fashion-MNIST."""
        self.dataset = mnist(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.test_init_op = self.dataset.test_init_op

        x, y = self.dataset.batch
        linear_outputs = _mlp(x, num_outputs=10)

        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs)

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.losses.get_regularization_loss()
