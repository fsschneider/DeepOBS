# -*- coding: utf-8 -*-
"""A simple N-Dimensional Noisy Quadratic Problem with Deep Learning eigenvalues."""

import numpy as np
from ._quadratic import _quadratic_base

# Random generator with a fixed seed to randomly draw eigenvalues and rotation.
# These are fixed properties of the test problem and should _not_ be randomized.
rng = np.random.RandomState(42)

def random_rotation(D):
    """Produces a rotation matrix R in SO(D) (the special orthogonal
    group SO(D), or orthogonal matrices with unit determinant, drawn uniformly
    from the Haar measure.
    The algorithm used is the subgroup algorithm as originally proposed by
    P. Diaconis & M. Shahshahani, "The subgroup algorithm for generating
    uniform random variables". Probability in the Engineering and
    Informational Sciences 1: 15?32 (1987)

    Args:
        D (int): Dimensionality of the matrix.

    Returns:
        np.array: Random rotation matrix ``R``.

    """
    assert D >= 2
    D = int(D)  # make sure that the dimension is an integer

    # induction start: uniform draw from D=2 Haar measure
    t = 2 * np.pi * rng.uniform()
    R = [[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]

    for d in range(2, D):
        v = rng.normal(size=(d + 1, 1))
        # draw on S_d the unit sphere
        v = np.divide(v, np.sqrt(np.transpose(v).dot(v)))
        e = np.concatenate((np.array([[1.0]]), np.zeros((d, 1))), axis=0)
        # random coset location of SO(d-1) in SO(d)
        x = np.divide((e - v), (np.sqrt(np.transpose(e - v).dot(e - v))))

        D = np.vstack([
            np.hstack([[[1.0]], np.zeros((1, d))]),
            np.hstack([np.zeros((d, 1)), R])
        ])
        R = D - 2 * np.outer(x, np.transpose(x).dot(D))
    # return negative to fix determinant
    return np.negative(R)


class quadratic_deep(_quadratic_base):
    r"""DeepOBS test problem class for a stochastic quadratic test problem ``100``\
    dimensions. The 90 % of the eigenvalues of the Hessian are drawn from the\
    interval :math:`(0.0, 1.0)` and the other 10 % are from :math:`(30.0, 60.0)` \
    simulating an eigenspectrum which has been reported for Deep Learning \
    https://arxiv.org/abs/1611.01838.

    This creatis a loss functions of the form

    :math:`0.5* (\theta - x)^T * Q * (\theta - x)`

    with Hessian ``Q`` and "data" ``x`` coming from the quadratic data set, i.e.,
    zero-mean normal.

    Args:
      batch_size (int): Batch size to use.
      weight_decay (float): No weight decay (L2-regularization) is used in this
          test problem. Defaults to ``None`` and any input here is ignored.

    Attributes:
      dataset: The DeepOBS data set class for the quadratic test problem.
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
        """Create a new quadratic deep test problem instance.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        eigenvalues = np.concatenate(
            (rng.uniform(0., 1., 90), rng.uniform(30., 60., 10)), axis=0)
        D = np.diag(eigenvalues)
        R = random_rotation(D.shape[0])
        hessian = np.matmul(np.transpose(R), np.matmul(D, R))
        super(quadratic_deep, self).__init__(batch_size, weight_decay, hessian)
