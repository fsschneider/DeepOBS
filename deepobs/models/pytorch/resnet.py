"""Residual network architectures for DeepOBS in PyTorch."""

from torch import nn

from deepobs.models.pytorch._utils import _ResidualBlock, _tfconv2d


class WRN(nn.Sequential):
    """A Wide Residual Network.

    Note: Proposed in

        - Sergey Zagoruyko, Nikos Komodakis
          Wide Residual Networks (2016).
    """

    def __init__(
        self, num_residual_blocks, widening_factor, num_outputs, bn_momentum=0.9
    ):
        """Build the network.

        Args:
            num_residual_blocks (int): Number of residual blocks.
            widening_factor (int): Widening factor of the network.
            num_outputs (int, optional): The numer of outputs (i.e. target classes).
                Defaults to ``10``.
            bn_momentum (float, optional): Momentum parameter of BatchNorm.
                Defaults to 0.9.
        """
        super(WRN, self).__init__()

        # initial conv
        self.add_module(
            "conv1", _tfconv2d(3, 16, 3, bias=False, tf_padding_type="same")
        )

        self._filters = [
            16,
            16 * widening_factor,
            32 * widening_factor,
            64 * widening_factor,
        ]
        self._strides = [1, 2, 2]

        # loop over three residual groups
        for group_number in range(1, 4):
            # first residual block is special since it has to change the number
            # of output channels for the skip connection.
            self.add_module(
                "res_unit" + str(group_number) + str(1),
                _ResidualBlock(
                    in_channels=self._filters[group_number - 1],
                    out_channels=self._filters[group_number],
                    first_stride=self._strides[group_number - 1],
                    is_first_block=True,
                ),
            )

            # loop over further residual blocks of this group
            for residual_block_number in range(1, num_residual_blocks):
                self.add_module(
                    "res_unit" + str(group_number) + str(residual_block_number + 1),
                    _ResidualBlock(
                        in_channels=self._filters[group_number],
                        out_channels=self._filters[group_number],
                    ),
                )
        # last layer
        self.add_module("bn", nn.BatchNorm2d(self._filters[3], momentum=bn_momentum))
        self.add_module("relu", nn.ReLU())
        self.add_module("avg_pool", nn.AvgPool2d(8))

        # reshape and dense layer
        self.add_module("flatten", nn.Flatten())
        self.add_module(
            "dense", nn.Linear(in_features=self._filters[3], out_features=num_outputs)
        )

        # initialisation
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)  # gamma
                nn.init.constant_(module.bias, 0.0)  # beta
                nn.init.constant_(module.running_mean, 0.0)
                nn.init.constant_(module.running_var, 1.0)
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
