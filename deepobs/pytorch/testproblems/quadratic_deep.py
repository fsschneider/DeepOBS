# -*- coding: utf-8 -*-
"""A simple N-Dimensional Noisy Quadratic Problem with Deep Learning eigenvalues."""

import numpy as np
from .testproblem import UnregularizedTestproblem
import torch
from .testproblems_modules import net_quadratic_deep
from ..datasets.quadratic import quadratic


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


class quadratic_deep(UnregularizedTestproblem):
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
        data: The DeepOBS data set class for the quadratic problem.
        loss_function: None. The output of the model is the loss.
        net: The DeepOBS subclass of torch.nn.Module that is trained for this tesproblem (net_quadratic_deep).
          """

    def __init__(self, batch_size, weight_decay=None):
        """Create a new quadratic deep test problem instance.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(quadratic_deep, self).__init__(batch_size, weight_decay)

    def quadratic_deep_loss_function_factory(self, reduction='mean'):
        def quadratic_deep_loss_function(inputs):
            batched_loss = self.net(inputs)
            if reduction == 'mean':
                return batched_loss.mean()
            elif reduction == 'sum':
                return torch.sum(batched_loss)
            elif reduction == 'none':
                return batched_loss
            else:
                raise NotImplementedError('Reduction ' + reduction + ' not implemented')
        return quadratic_deep_loss_function

    def set_up(self):
        rng = np.random.RandomState(42)
        eigenvalues = np.concatenate(
            (rng.uniform(0., 1., 90), rng.uniform(30., 60., 10)), axis=0)
        D = np.diag(eigenvalues)
        R = random_rotation(D.shape[0])
        Hessian = np.matmul(np.transpose(R), np.matmul(D, R))
        Hessian = torch.from_numpy(Hessian).to(self._device, torch.float32)
        self.net = net_quadratic_deep(100, Hessian)

        self.data = quadratic(self._batch_size)
        self.net.to(self._device)
        self.loss_function = self.quadratic_deep_loss_function_factory
        self.regularization_groups = self.get_regularization_groups()

    def get_batch_loss_and_accuracy(self, return_forward_func = False, reduction='mean',
                                    add_regularization_if_available=True):
        """Gets a new batch and calculates the loss and accuracy (if available)
        on that batch. This is a default implementation for image classification.
        Testproblems with different calculation routines (e.g. RNNs) overwrite this method accordingly.

        Args:
            return_forward_func (bool): If ``True``, the call also returns a function that calculates the loss on the current batch. Can be used if you need to access the forward path twice.
        Returns:
            float, float, (callable): loss and accuracy of the model on the current batch. If ``return_forward_func`` is ``True`` it also returns the function that calculates the loss on the current batch.
            """
        inputs = self._get_next_batch()[0]
        inputs = inputs.to(self._device)
        def _get_batch_loss_and_accuracy():
            # in evaluation phase is no gradient needed
            if self.phase in ["train_eval", "test", "valid"]:
                with torch.no_grad():
                    loss = self.loss_function(reduction=reduction)(inputs)
            else:
                loss = self.loss_function(reduction=reduction)(inputs)

            accuracy = 0.0

            if add_regularization_if_available:
                regularizer_loss = self.get_regularization_loss()
            else:
                torch.tensor(0.0, device=torch.device(self._device))

            return loss + regularizer_loss, accuracy

        if return_forward_func:
            return _get_batch_loss_and_accuracy(), _get_batch_loss_and_accuracy
        else:
            return _get_batch_loss_and_accuracy()