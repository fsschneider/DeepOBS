# -*- coding: utf-8 -*-
"""VGG 19 architecture for CIFAR-100."""

import tensorflow as tf

from ..datasets.cifar100 import cifar100
from ._vgg import _vgg
from .testproblem import TestProblem


class cifar100_vgg19(TestProblem):
    """DeepOBS test problem class for the VGG 19 network on Cifar-100.

  The CIFAR-100 images are resized to ``224`` by ``224`` to fit the input
  dimension of the original VGG network, which was designed for ImageNet.

  Details about the architecture can be found in the `original paper`_.
  VGG 19 consists of 19 weight layers, of mostly convolutions. The model uses
  cross-entroy loss. L2-Regularization is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  .. _original paper: https://arxiv.org/abs/1409.1556

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
        is used on the weights but not the biases.
        Defaults to ``5e-4``.

  Attributes:
    dataset: The DeepOBS data set class for Cifar-100.
    train_init_op: A tensorflow operation initializing the test problem for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the test problem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the test problem for
        evaluating on test data.
    losses: A tf.Tensor of shape (batch_size, ) containing the per-example loss
        values.
    regularizer: A scalar tf.Tensor containing a regularization term.
    accuracy: A scalar tf.Tensor containing the mini-batch mean accuracy.
  """

    def __init__(self, batch_size, l2_reg=0.0005):
        """Create a new VGG 19 test problem instance on Cifar-100.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(cifar100_vgg19, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the VGG 19 test problem on Cifar-100."""
        self.dataset = cifar100(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        training = tf.equal(self.dataset.phase, "train")
        x, y = self.dataset.batch
        linear_outputs = _vgg(
            x, training, variant=19, num_outputs=100, l2_reg=self._l2_reg,
        )

        self.losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=linear_outputs
        )
        y_pred = tf.argmax(input=linear_outputs, axis=1)
        y_correct = tf.argmax(input=y, axis=1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.compat.v1.losses.get_regularization_loss()
