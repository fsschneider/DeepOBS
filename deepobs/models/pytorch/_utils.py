"""Utility functions for the network architectures for DeepOBS in PyTorch."""

from math import ceil

import torch
from numpy.random import RandomState
from scipy.stats import truncnorm as tn
from torch import nn


def _tfconv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    tf_padding_type=None,
):
    """Adds ``padding_type=same`` to PyTorch convolutions, to mimic TF."""
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = _hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )
    return nn.Sequential(*modules)


def _tfconv2d_transpose(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    output_padding=0,
    groups=1,
    bias=True,
    dilation=1,
    tf_padding_type=None,
):
    """Adds ``padding_type=same`` to PyTorch transpose convolutions, to mimic TF."""
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = _hook_factory_tf_inverse_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    # eliminate the effect of the in-build padding (is not capable of asymmeric padding)
    if isinstance(kernel_size, int):
        padding = kernel_size - 1
    else:
        padding = (kernel_size[0] - 1, kernel_size[1] - 1)

    modules.append(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
        )
    )

    return nn.Sequential(*modules)


def _tfmaxpool2d(
    kernel_size,
    stride=None,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    tf_padding_type=None,
):
    """Adds ``padding_type=same`` to PyTorch maxpooling, to mimic TF."""
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = _hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(
        nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
    )

    return nn.Sequential(*modules)


def _truncated_normal_init(tensor, mean=0.0, stddev=1.0):
    """A truncated normal initialization method for PyTorch (mimicing TF).

    Args:
        tensor (torch.Tensor): The tensor to init.
        mean (float, optional): The mean of the normal distribution. Defaults to 0.
        stddev (float, optional): Stddev of the normal distribution. Defaults to 1.

    Returns:
        torch.Tensor: Initialized tensor.
    """
    total_size = tensor.numel()

    # determine the scipy random state from the torch seed
    # the numpy seed can be between 0 and 2**32-1
    np_seed = torch.randint(0, 2 ** 32 - 1, (1, 1)).view(-1).item()
    np_state = RandomState(np_seed)
    # truncates 2 std from mean, since rescaling: a = ((mean-2std)-mean)/std = -2
    samples = tn.rvs(
        a=-2,
        b=2,
        loc=mean,
        scale=stddev,
        size=total_size,
        random_state=np_state,
    )
    samples = samples.reshape(tuple(tensor.size()))
    init_tensor = torch.from_numpy(samples).type_as(tensor)
    return init_tensor


def _mean_allcnnc():
    """The all convolution layer implementation of torch.mean()."""
    # TODO implement pre forward hook to adapt to arbitary image size for other
    # data sets than cifar100
    return nn.Sequential(nn.AvgPool2d(kernel_size=(6, 6)), nn.Flatten())


class _ResidualBlock(nn.Module):
    """A residual block, the main component of the wide residual net.

    Note: Proposed in

        - Sergey Zagoruyko, Nikos Komodakis
          Wide Residual Networks (2016).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        first_stride=1,
        is_first_block=False,
        bn_momentum=0.9,
    ):
        """Initialize the residual block.

        Args:
            in_channels (int): Number of in_channels.
            out_channels (int): Number of out_channels.
            kernel_size (int, optional): Size of the kernel. Defaults to 3.
            first_stride (int, optional): First stride. Defaults to 1.
            is_first_block (bool, optional): Whether or not it is the first block.
                Defaults to False.
            bn_momentum (float, optional): Momentum value of BatchNorm.
                Defaults to 0.9.
        """
        super(_ResidualBlock, self).__init__()

        self.is_first_block = is_first_block

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        self.relu1 = nn.ReLU()

        if self.is_first_block:
            self.convFirstBlock = _tfconv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=first_stride,
                tf_padding_type="same",
                bias=False,
            )

        self.conv1 = _tfconv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=first_stride,
            tf_padding_type="same",
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.relu2 = nn.ReLU()
        self.conv2 = _tfconv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            tf_padding_type="same",
            bias=False,
        )

    def forward(self, x):
        """Forward pass."""
        if self.is_first_block:
            x = self.bn1(x)
            x = self.relu1(x)
            identity = self.convFirstBlock(x)
        else:
            identity = x
            x = self.bn1(x)
            x = self.relu1(x)

        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x += identity

        return x


def _hook_factory_tf_padding_same(kernel_size, stride):
    """Generates the pre-forward hook for PyTorch convs to add padding ``same``."""

    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_padding_from_tf_same(
            image_dimensions, kernel_size, stride
        )

    return hook


def _hook_factory_tf_inverse_padding_same(kernel_size, stride):
    """Generates the pre-forward hook for transpose_convs to add padding ``same``."""

    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_inverse_padding_from_tf_same(
            image_dimensions, kernel_size, stride
        )

    return hook


def _determine_padding_from_tf_same(
    input_dimensions, kernel_dimensions, stride_dimensions
):
    """Implements TF's padding 'same' for kernel processes like convs or pooling.

    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution

    Returns:
        tuple: A 4-tuple for padding layer creation that mimics tf's padding 'same'.
    """
    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = ceil(in_height / stride_height)
    out_width = ceil(in_width / stride_width)

    # determine the pad size along each dimension
    pad_along_height = max(
        (out_height - 1) * stride_height + kernel_height - in_height, 0
    )
    pad_along_width = max((out_width - 1) * stride_width + kernel_width - in_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


def _determine_inverse_padding_from_tf_same(
    input_dimensions, kernel_dimensions, stride_dimensions
):
    """Implements tf's padding 'same' for inverse processes such as transpose convs.

    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution

    Returns:
        tuple: 4-tuple for padding layer creation that mimics tf's padding 'same'.
    """
    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = in_height * stride_height
    out_width = in_width * stride_width

    # determine the pad size along each dimension
    pad_along_height = max(
        (in_height - 1) * stride_height + kernel_height - out_height, 0
    )
    pad_along_width = max((in_width - 1) * stride_width + kernel_width - out_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom
