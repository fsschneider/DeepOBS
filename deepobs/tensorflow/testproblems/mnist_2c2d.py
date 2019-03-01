# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for MNIST."""

import tensorflow as tf

from ._2c2d import _2c2d
from ..datasets.mnist import mnist
from .testproblem import TestProblem


class mnist_2c2d(TestProblem):
    """DeepOBS test problem class for a two convolutional and two dense layered\
    neural network on MNIST.

  The network has been adapted from the `TensorFlow tutorial\
  <https://www.tensorflow.org/tutorials/estimators/cnn>`_ and consists of

    - two conv layers with ReLUs, each followed by max-pooling
    - one fully-connected layers with ReLUs
    - 10-unit output layer with softmax
    - cross-entropy loss
    - No regularization

  The weight matrices are initialized with truncated normal (standard deviation
  of ``0.05``) and the biases are initialized to ``0.05``.

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): No weight decay (L2-regularization) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

  Attributes:
    dataset: The DeepOBS data set class for MNIST.
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
        """Create a new 2c2d test problem instance on MNIST.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(mnist_2c2d, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )


    def set_up(self):
        """Sets up the vanilla CNN test problem on MNIST."""
        self.dataset = mnist(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.test_init_op = self.dataset.test_init_op

        x, y = self.dataset.batch
        linear_outputs = _2c2d(
            x,
            num_outputs=10)

        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs)

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.losses.get_regularization_loss()
