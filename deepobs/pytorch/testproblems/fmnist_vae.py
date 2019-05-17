# -*- coding: utf-8 -*-
"""A VAE architecture for Fashion MNIST."""

import torch
from .testproblems_modules import net_vae
from ..datasets.fmnist import fmnist
from .testproblem import TestProblem


class fmnist_vae(TestProblem):

    def __init__(self, batch_size, weight_decay=None):

        super(fmnist_vae, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )

        def loss_function(outputs, targets, mean, std_dev):
            outputs_flat = outputs.view(-1, 28*28)
            targets_flat = targets.view(-1, 28*28)
            image_loss = torch.mean((outputs_flat - targets_flat).pow(2).sum(dim=1))
            latent_loss = -0.5 * torch.mean((1 + 2 * std_dev - mean.pow(2) - torch.exp(2*std_dev)).sum(dim=1))
            return image_loss + latent_loss

        self.loss_function=loss_function


    def set_up(self):
        self.data = fmnist(self._batch_size)
        self.net = net_vae(n_latent = 8)
        self.net.to(self._device)


    def get_batch_loss_and_accuracy(self):
        inputs, _ = self._get_next_batch()
        inputs = inputs.to(self._device)

        # in evaluation phase is no gradient needed
        if self.phase in ["train_eval", "test"]:
            with torch.no_grad():
                outputs, means, std_devs = self.net(inputs)
                loss = self.loss_function(outputs, inputs, means, std_devs)
        else:
            outputs, means, std_devs = self.net(inputs)
            loss = self.loss_function(outputs, inputs, means, std_devs)


        accuracy = 0
        return loss, accuracy

