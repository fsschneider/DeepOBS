# -*- coding: utf-8 -*-
"""Utility functions for running optimizers."""

import numpy as np
import torchvision.models as models

from torch import nn
from torch.optim import lr_scheduler
from numpy import asarray, cov, trace, iscomplexobj
from numpy.random import shuffle
from scipy.linalg import sqrtm
from skimage.transform import resize



def make_lr_schedule(optimizer, lr_sched_epochs=None, lr_sched_factors=None):
    """Creates a learning rate schedule in the form of a torch.optim.lr_scheduler.LambdaLR instance.

  After ``lr_sched_epochs[i]`` epochs of training, the learning rate will be set
  to ``lr_sched_factors[i] * lr_base``.

  Examples:
    - ``make_schedule(optim.SGD(net.parameters(), lr = 0.5), [50, 100], [0.1, 0.01])`` yields
      to the following schedule for the SGD optimizer on the parameters of net:
      SGD uses lr = 0.5 for epochs 0 to 49.
      SGD uses lr = 0.5*0.1 = 0.05 for epochs 50 to 99.
      SGD uses lr = 0.5*0.01 = 0.005 for epochs 100 to end.

  Args:
    optimizer: The optimizer for which the schedule is set. It already holds the base learning rate.
    lr_sched_epochs: A list of integers, specifying epochs at
        which to decrease the learning rate.
    lr_sched_factors: A list of floats, specifying factors by
        which to decrease the learning rate.

  Returns:
    sched: A torch.optim.lr_scheduler.LambdaLR instance with a function that determines the learning rate at every epoch.
  """

    if (lr_sched_factors is None) or (lr_sched_epochs is None):
        determine_lr = lambda epoch: 1
    else:

        def determine_lr(epoch):
            if epoch < lr_sched_epochs[0]:
                return 1
            else:
                help_array = np.array(lr_sched_epochs)
                index = np.argmax(np.where(help_array <= epoch)[0])
                return lr_sched_factors[index]

    sched = lr_scheduler.LambdaLR(optimizer, determine_lr)
    return sched


class gan_eval_inception(nn.Module):
    def __init__(self, img_list):
        super(gan_eval_inception, self).__init__()

   # inception = models.inception_v3(pretrained=True, progress=True)

    # scale an array of images to a new size
    def scale_images(img_list, new_shape):
        new_img_list = []
        for image in img_list:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            new_img_list.append(new_image)
        return asarray(new_img_list)

    # calculate frechet inception distance
    def calculate_fid(images1, images2):
        # calculate activations
        act1 = gan_eval_inception.inception.predict(images1)
        act2 = gan_eval_inception.inception.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    pass

