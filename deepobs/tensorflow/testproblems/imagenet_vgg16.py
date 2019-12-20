# -*- coding: utf-8 -*-
"""VGG 16 architecture for ImageNet."""

import tensorflow as tf

from ..datasets.imagenet import imagenet
from ._vgg import _vgg
from .testproblem import TestProblem


class imagenet_vgg16(TestProblem):
    """DeepOBS test problem class for the VGG 16 network on ImageNet.

  Details about the architecture can be found in the `original paper`_.
  VGG 16 consists of 16 weight layers, of mostly convolutions. The model uses
  cross-entroy loss. L2-Regularization is used on the weights (but not the biases)
  which defaults to ``5e-4``.

  .. _original paper: https://arxiv.org/abs/1409.1556

  Args:
    batch_size (int): Batch size to use.
    l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
        is used on the weights but not the biases.
        Defaults to ``5e-4``.

  Attributes:
    dataset: The DeepOBS data set class for ImageNet.
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

    def __init__(self, batch_size, l2_reg=5e-4):
        """Create a new VGG 16 test problem instance on ImageNet.

        Args:
          batch_size (int): Batch size to use.
          l2_reg (float): L2-regularization factor. L2-Regularization (weight decay)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(imagenet_vgg16, self).__init__(batch_size, l2_reg)

    def set_up(self):
        """Set up the VGG 16 test problem on ImageNet."""
        self.dataset = imagenet(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        training = tf.equal(self.dataset.phase, "train")
        x, y = self.dataset.batch
        linear_outputs = _vgg(
            x,
            training,
            variant=16,
            num_outputs=1001,
            l2_reg=self._l2_reg,
        )

        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=linear_outputs
        )
        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.losses.get_regularization_loss()
