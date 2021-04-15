"""Variational autoencoder network architectures for DeepOBS in PyTorch."""

import torch
from torch import nn

from deepobs.models.pytorch._utils import _tfconv2d, _tfconv2d_transpose


class VAE(nn.Module):
    r"""A basic Variational Autoencode (VAE).

    The network has been adapted from the `here
    <https://towardsdatascience.com/teaching-a-variational-autoencoder-\
        vae-to-draw-mnist-characters-978675c95776>`_
    and consists of an encoder:
    - With three convolutional layers with each ``64`` filters.
    - Using a leaky ReLU activation function with :math:`\\alpha = 0.3`
    - Dropout layers after each convolutional layer with a rate of ``0.2``.
    and an decoder:
    - With two dense layers with ``24`` and ``49`` units and leaky ReLU activation.
    - With three deconvolutional layers with each ``64`` filters.
    - Dropout layers after the first two deconvolutional layer with a rate of ``0.2``.
    - A final dense layer with ``28 x 28`` units and sigmoid activation.
    """

    def __init__(self, n_latent):
        """Build the network.

        Args:
            n_latent (int): Size of the latent space.
        """
        super(VAE, self).__init__()
        self.n_latent = n_latent

        # encoding layers
        self.conv1 = _tfconv2d(1, 64, kernel_size=4, stride=2, tf_padding_type="same")
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = _tfconv2d(64, 64, kernel_size=4, stride=2, tf_padding_type="same")
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = _tfconv2d(64, 64, kernel_size=4, stride=1, tf_padding_type="same")
        self.dropout3 = nn.Dropout(p=0.2)

        self.dense1 = nn.Linear(in_features=7 * 7 * 64, out_features=self.n_latent)
        self.dense2 = nn.Linear(in_features=7 * 7 * 64, out_features=self.n_latent)

        # decoding layers
        self.dense3 = nn.Linear(in_features=self.n_latent, out_features=24)
        self.dense4 = nn.Linear(in_features=24, out_features=24 * 2 + 1)

        self.deconv1 = _tfconv2d_transpose(
            1, 64, kernel_size=4, stride=2, tf_padding_type="same"
        )
        self.dropout4 = nn.Dropout(p=0.2)

        self.deconv2 = _tfconv2d_transpose(
            64, 64, kernel_size=4, stride=1, tf_padding_type="same"
        )
        self.dropout5 = nn.Dropout(p=0.2)

        self.deconv3 = _tfconv2d_transpose(
            64, 64, kernel_size=4, stride=1, tf_padding_type="same"
        )
        self.dropout6 = nn.Dropout(p=0.2)

        self.dense5 = nn.Linear(in_features=14 * 14 * 64, out_features=28 * 28)

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)

    def encode(self, x):
        """Build the encoder of the VAE.

        Args:
            x (torch.Tensor): Input to the encoder.

        Returns:
            tuple: `z`, `mean` and `std_dev` output of the encoder.
        """
        x = nn.functional.leaky_relu(self.conv1(x), negative_slope=0.3)
        x = self.dropout1(x)

        x = nn.functional.leaky_relu(self.conv2(x), negative_slope=0.3)
        x = self.dropout2(x)

        x = nn.functional.leaky_relu(self.conv3(x), negative_slope=0.3)
        x = self.dropout3(x)

        x = x.view(-1, 7 * 7 * 64)

        mean = self.dense1(x)
        std_dev = 0.5 * self.dense2(x)
        eps = torch.randn_like(std_dev)
        z = mean + eps * torch.exp(std_dev)

        return z, mean, std_dev

    def decode(self, z):
        """Build the decoder of the VAE.

        Args:
            z (torch.Tensor): Input to the decoder.

        Returns:
            torch.Tensor: A batch of created images.
        """
        x = nn.functional.leaky_relu(self.dense3(z), negative_slope=0.3)
        x = nn.functional.leaky_relu(self.dense4(x), negative_slope=0.3)

        x = x.view(-1, 1, 7, 7)

        x = nn.functional.relu(self.deconv1(x))
        x = self.dropout4(x)

        x = nn.functional.relu(self.deconv2(x))
        x = self.dropout5(x)

        x = nn.functional.relu(self.deconv3(x))
        x = self.dropout6(x)

        x = x.view(-1, 14 * 14 * 64)

        x = nn.functional.sigmoid(self.dense5(x))

        images = x.view(-1, 1, 28, 28)

        return images

    def forward(self, x):
        """Build forward pass of the model.

        Args:
            x (torch.Tensor): Input to the encoder

        Returns:
            tuple: `image`, `mean` and `std_dev` of the VAE.
        """
        z, mean, std_dev = self.encode(x)

        image = self.decode(z)

        return image, mean, std_dev
