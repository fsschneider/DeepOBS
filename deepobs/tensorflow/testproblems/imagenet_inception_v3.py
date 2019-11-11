# -*- coding: utf-8 -*-
"""Inception version 3 architecture for ImageNet."""

import tensorflow as tf

from ._inception_v3 import _inception_v3
from ..datasets.imagenet import imagenet
from .testproblem import TestProblem


class imagenet_inception_v3(TestProblem):
    """DeepOBS test problem class for the Inception version 3 architecture on
  ImageNet.

  Details about the architecture can be found in the `original paper`_.

  There are many changes from the paper to the `official Tensorflow implementation\
  <https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/inception_model.py>`_
  as well as the model.txt that can be found in the sources of the original
  paper. We chose to implement the version from Tensorflow (with possibly some
  minor changes)

  In the `original paper`_ they trained the network using:

  - ``100`` Epochs.
  - Batch size ``32``.
  - RMSProp with a decay of ``0.9`` and :math:`\\epsilon = 1.0`.
  - Initial learning rate ``0.045``.
  - Learning rate decay every two epochs with exponential rate of ``0.94``.
  - Gradient clipping with threshold 2.0

  .. _original paper: https://arxiv.org/abs/1512.00567

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
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

    def __init__(self, batch_size, weight_decay=5e-4):
        """Create a new Inception v3 test problem instance on ImageNet.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
              is used on the weights but not the biases.
              Defaults to ``5e-4``.
        """
        super(imagenet_inception_v3, self).__init__(batch_size, weight_decay)

    def set_up(self):
        """Set up the Inception v3 test problem on ImageNet."""
        self.dataset = imagenet(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op
        training = tf.equal(self.dataset.phase, "train")
        x, y = self.dataset.batch

        linear_outputs, aux_linear_outputs = _inception_v3(
            x, training, weight_decay=self._weight_decay
        )

        # Compute two components of losses
        # reduction=tf.losses.Reduction.None means output will have size
        # ``batch_size``
        aux_losses = tf.losses.softmax_cross_entropy(
            onehot_labels=y,
            logits=aux_linear_outputs,
            weights=0.4,
            label_smoothing=0.1,
            reduction=tf.losses.Reduction.NONE,
        )
        main_losses = tf.losses.softmax_cross_entropy(
            onehot_labels=y,
            logits=linear_outputs,
            label_smoothing=0.1,
            reduction=tf.losses.Reduction.NONE,
        )

        # Add main_loss and aux_loss if we are training
        self.losses = tf.cond(
            training,
            lambda: tf.add(main_losses, aux_losses),
            lambda: tf.add(main_losses, 0.0),
            name="losses",
        )

        y_pred = tf.argmax(linear_outputs, 1)
        y_correct = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_pred, y_correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.losses.get_regularization_loss()
