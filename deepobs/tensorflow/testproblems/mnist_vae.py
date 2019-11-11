# -*- coding: utf-8 -*-
"""A Variational Autoencoder architecture for MNIST."""

import tensorflow as tf

from ._vae import _vae
from ..datasets.mnist import mnist
from .testproblem import TestProblem


class mnist_vae(TestProblem):
    """DeepOBS test problem class for a variational autoencoder (VAE) on MNIST.

  The network has been adapted from the `here\
  <https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776>`_
  and consists of an encoder:

    - With three convolutional layers with each ``64`` filters.
    - Using a leaky ReLU activation function with :math:`\\alpha = 0.3`
    - Dropout layers after each convolutional layer with a rate of ``0.2``.

  and an decoder:

    - With two dense layers with ``24`` and ``49`` units and leaky ReLU activation.
    - With three deconvolutional layers with each ``64`` filters.
    - Dropout layers after the first two deconvolutional layer with a rate of ``0.2``.
    - A final dense layer with ``28 x 28`` units and sigmoid activation.

  No regularization is used.

  Args:
      batch_size (type): Batch size to use.
      weight_decay (type): No weight decay (L2-regularization) is used in this
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
  """

    def __init__(self, batch_size, weight_decay=None):
        """Create a new VAE test problem instance on MNIST.

        Args:
            batch_size (type): Batch size to use.
            weight_decay (type): No weight decay (L2-regularization) is used in this
                test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(mnist_vae, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model.",
            )

    def set_up(self):
        """Sets up the VAE test problem on MNIST."""
        self.dataset = mnist(self._batch_size)
        self.train_init_op = self.dataset.train_init_op
        self.train_eval_init_op = self.dataset.train_eval_init_op
        self.valid_init_op = self.dataset.valid_init_op
        self.test_init_op = self.dataset.test_init_op

        training = tf.equal(self.dataset.phase, "train")
        x, _ = self.dataset.batch
        img, mean, std_dev = _vae(x, training, n_latent=8)

        # Define Loss
        flatten_img = tf.reshape(img, [-1, 28 * 28])
        x_flat = tf.reshape(x, shape=[-1, 28 * 28])
        img_loss = tf.reduce_sum(tf.squared_difference(flatten_img, x_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1.0 + 2.0 * std_dev - tf.square(mean) - tf.exp(2.0 * std_dev), 1
        )
        self.losses = img_loss + latent_loss

        self.regularizer = tf.losses.get_regularization_loss()
