#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from scipy.stats import truncnorm as tn
from math import ceil
from torch import nn
from torch.nn import functional as F

def _determine_inverse_padding_from_tf_same(input_dimensions, kernel_dimensions, stride_dimensions):
    # implements tf's padding 'same'
    # input: dimensions are tuple (height, width) or ints for quadratic dimensions
    # output: a padding 4-tuple for padding layer creation that mimics tf's padding 'same'

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
    pad_along_height = max((in_height - 1) * stride_height + kernel_height - out_height, 0)
    pad_along_width = max((in_width - 1) * stride_width + kernel_width - out_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)

def _determine_padding_from_tf_same(input_dimensions, kernel_dimensions, stride_dimensions):
    # implements tf's padding 'same'
    # input: dimensions are tuple (height, width) or ints for quadratic dimensions
    # output: a padding 4-tuple for padding layer creation that mimics tf's padding 'same'

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
    pad_along_height = max((out_height - 1) * stride_height + kernel_height - in_height, 0)
    pad_along_width = max((out_width - 1) * stride_width + kernel_width - in_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)

def _truncated_normal_init(tensor, mean=0, stddev=1):
    total_size = tensor.numel()
    # truncates 2 std from mean, since rescaling: a = ((mean-2std)-mean)/std = -2
    samples = tn.rvs(a = -2, b = 2, loc = mean, scale = stddev, size = total_size)
    samples = samples.reshape(tuple(tensor.size()))
    init_tensor = torch.from_numpy(samples).type_as(tensor)
    return init_tensor

class tfmaxpool2d(nn.MaxPool2d):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False,
                 tf_padding_type = None):

        super(tfmaxpool2d, self).__init__(kernel_size,
                                             stride,
                                             padding,
                                             dilation,
                                             return_indices,
                                             ceil_mode)

        self.tf_padding_type = tf_padding_type

    def forward(self, x):
        if self.tf_padding_type == 'same':
            with torch.no_grad():
                image_dimensions = x.size()[-2:]
                padding_4_tupel = _determine_padding_from_tf_same(image_dimensions, self.kernel_size, self.stride)
            x = F.pad(x, padding_4_tupel)
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)


class tfconv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 tf_padding_type = None):

        super(tfconv2d, self).__init__(in_channels,
             out_channels,
             kernel_size,
             stride,
             padding,
             dilation,
             groups,
             bias)

        self.tf_padding_type = tf_padding_type

    def forward(self, x):
        if self.tf_padding_type == 'same':
            with torch.no_grad():
                image_dimensions = x.size()[-2:]
                padding_4_tupel = _determine_padding_from_tf_same(image_dimensions, self.kernel_size, self.stride)
            x = F.pad(x, padding_4_tupel)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class tfconv2d_transpose(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding = 0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 tf_padding_type = None):

        super(tfconv2d_transpose, self).__init__(in_channels,
             out_channels,
             kernel_size,
             stride,
             padding,
             output_padding,
             groups,
             bias,
             dilation)

        self.tf_padding_type = tf_padding_type

    def forward(self, x):
        if self.tf_padding_type == 'same':
            # eliminate the effect of the in-build padding (is not capable of asymmeric padding)
            if isinstance(self.kernel_size, int):
                self.padding = self.kernel_size - 1
            else:
                self.padding = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
            padding_4_tupel = _determine_inverse_padding_from_tf_same(x.size()[-2:], self.kernel_size, self.stride)
            x = F.pad(x, padding_4_tupel)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

class flatten(nn.Module):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class mean_allcnnc(nn.Module):
    def __init__(self):
        super(mean_allcnnc, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2,3))

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, first_stride=1, is_first_block = False, bn_momentum = 0.9):
        super(residual_block, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels, momentum = bn_momentum)
        self.relu1 = nn.ReLU()
        self.conv1 = tfconv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = first_stride, tf_padding_type='same', bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.relu2 = nn.ReLU()
        self.conv2 = tfconv2d(in_channels = out_channels, out_channels=out_channels, kernel_size=kernel_size, stride = 1, tf_padding_type='same', bias=False)

        self.is_first_block = is_first_block

    def forward(self, x):

        if self.is_first_block:
            identity = self.conv1(x)
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