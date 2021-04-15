"""Generative adversarial network architectures for DeepOBS in PyTorch."""

from torch import nn


class DCGAN_G(nn.Module):
    """The generator network of a DCGAN.

    The network has been adapted from `here
    <https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>`
    """

    def __init__(self, num_channels, n_latent=100, fm_size=64):
        """Build the DCGAN generator.

        Args:
            num_channels (int): Number of color channels of the generated images.
                Color images have 3, MNIST-like images 1.
            n_latent (int, optional): Length of the latent vector. Defaults to ``100``.
            fm_size (int, optional): Size of the feature maps. Defaults to ``64``.
        """
        super(DCGAN_G, self).__init__()
        self.n_latent = n_latent
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                n_latent,
                fm_size * 8,
                4,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(fm_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                fm_size * 8,
                fm_size * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(fm_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                fm_size * 4,
                fm_size * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(fm_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                fm_size * 2,
                fm_size,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(fm_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                fm_size,
                num_channels,
                4,
                2,
                1,
                bias=False,
            ),
            nn.Tanh(),
        )

        # initialisation
        def weights_init(m):
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.ReLU, nn.Tanh, nn.Sequential)):
                pass
            else:
                raise TypeError

        self.main.apply(weights_init)

    def forward(self, input):
        """Forward pass of the generator."""
        return self.main(input)


class DCGAN_D(nn.Module):
    """The discriminator network of a DCGAN.

    The network has been adapted from `here
    <https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>`
    """

    def __init__(self, num_channels, fm_size=64):
        """Build the DCGAN discriminator.

        Args:
            num_channels (int): Number of color channels of the generated images.
                Color images have 3, MNIST-like images 1.
            fm_size (int, optional): Size of the feature maps. Defaults to ``64``.
        """
        super(DCGAN_D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                num_channels,
                fm_size,
                4,
                2,
                1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                fm_size,
                fm_size * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(fm_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (fm_size*2) x 16 x 16
            nn.Conv2d(
                fm_size * 2,
                fm_size * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(fm_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                fm_size * 4,
                fm_size * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(fm_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                fm_size * 8,
                1,
                4,
                1,
                0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        # initialisation
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LeakyReLU, nn.Sigmoid, nn.Sequential)):
                pass
            else:
                raise TypeError

        self.main.apply(weights_init)

    def forward(self, input):
        """Forward pass of the discriminator."""
        return self.main(input)
