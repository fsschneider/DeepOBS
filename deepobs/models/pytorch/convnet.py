"""Convolutional network architectures for DeepOBS in PyTorch."""

from torch import nn

from deepobs.models.pytorch._utils import (
    _mean_allcnnc,
    _tfconv2d,
    _tfmaxpool2d,
    _truncated_normal_init,
)

#  TODO Inception v3


class Basic2c2d(nn.Sequential):
    """Basic ConvNet with two conv and two dense layers.

    The network has been adapted from the `TensorFlow tutorial
    <https://www.tensorflow.org/tutorials/estimators/cnn>`_ and consists of
    - two conv layers with ReLUs, each followed by max-pooling
    - one fully-connected layers with ReLUs
    - output layer with softmax
    The weight matrices are initialized with truncated normal (standard deviation
    of ``0.05``) and the biases are initialized to ``0.05``.
    """

    def __init__(self, num_outputs):
        """Build the network.

        Args:
            num_outputs (int, optional): The numer of outputs (i.e. target classes).
                Defaults to ``10``.
        """
        super(Basic2c2d, self).__init__()
        self.add_module("conv1", _tfconv2d(1, 32, 5, tf_padding_type="same"))
        self.add_module("relu1", nn.ReLU())
        self.add_module("max_pool1", _tfmaxpool2d(2, stride=2, tf_padding_type="same"))

        self.add_module("conv2", _tfconv2d(32, 64, 5, tf_padding_type="same"))
        self.add_module("relu2", nn.ReLU())
        self.add_module("max_pool2", _tfmaxpool2d(2, stride=2, tf_padding_type="same"))

        self.add_module("flatten", nn.Flatten())

        self.add_module("dense1", nn.Linear(in_features=7 * 7 * 64, out_features=1024))
        self.add_module("relu3", nn.ReLU())

        self.add_module("dense2", nn.Linear(in_features=1024, out_features=num_outputs))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.05)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=0.05
                )

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.05)
                module.weight.data = _truncated_normal_init(
                    module.weight.data, mean=0, stddev=0.05
                )


class Basic3c3d(nn.Sequential):
    """Basic ConvNet with three conv and three dense layers.

    The network has been adapted from the `TensorFlow tutorial
    <https://www.tensorflow.org/tutorials/estimators/cnn>`_ and consists of
    - three conv layers with ReLUs, each followed by max-pooling
    - two fully-connected layers with ``512`` and ``256`` units and ReLU activation
    - output layer with softmax
    The weight matrices are initialized using Xavier initialization and the biases
    are initialized to ``0.0``
    """

    def __init__(self, num_outputs):
        """Build the network.

        Args:
            num_outputs (int, optional): The numer of outputs (i.e. target classes).
                Defaults to ``10``.
        """
        super(Basic3c3d, self).__init__()

        self.add_module("conv1", _tfconv2d(3, 64, 5))
        self.add_module("relu1", nn.ReLU())
        self.add_module("maxpool1", _tfmaxpool2d(3, stride=2, tf_padding_type="same"))

        self.add_module("conv2", _tfconv2d(64, 96, 3))
        self.add_module("relu2", nn.ReLU())
        self.add_module("maxpool2", _tfmaxpool2d(3, stride=2, tf_padding_type="same"))

        self.add_module("conv3", _tfconv2d(96, 128, 3, tf_padding_type="same"))
        self.add_module("relu3", nn.ReLU())
        self.add_module("maxpool3", _tfmaxpool2d(3, stride=2, tf_padding_type="same"))

        self.add_module("flatten", nn.Flatten())

        self.add_module("dense1", nn.Linear(in_features=3 * 3 * 128, out_features=512))
        self.add_module("relu4", nn.ReLU())

        self.add_module("dense2", nn.Linear(in_features=512, out_features=256))
        self.add_module("relu5", nn.ReLU())

        self.add_module("dense3", nn.Linear(in_features=256, out_features=num_outputs))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)


class VGG(nn.Sequential):
    """Classic VGG network.

    Note: Proposed in

        - Xiangyu Zhang, Jianhua Zou, Kaiming He, Jian Sun
          Accelerating Very Deep Convolutional Networks for Classification and
          Detection (2015).
    """

    def __init__(self, num_outputs, variant):
        """Build the network.

        Args:
            num_outputs (int, optional): The numer of outputs (i.e. target classes).
                Defaults to ``100``.
            variant (int, optional): The variant of VGG network to use. Can be
            one of ``16`` or ``19``. Defaults to ``16``.
        """
        super(VGG, self).__init__()

        assert variant == 16 or variant == 19

        self.add_module("upsampling", nn.UpsamplingBilinear2d(size=(224, 224)))

        self.add_module("conv11", _tfconv2d(3, 64, 3, tf_padding_type="same"))
        self.add_module("relu11", nn.ReLU())

        self.add_module("conv12", _tfconv2d(64, 64, 3, tf_padding_type="same"))
        self.add_module("relu12", nn.ReLU())
        self.add_module("max_pool1", _tfmaxpool2d(2, stride=2, tf_padding_type="same"))

        self.add_module("conv21", _tfconv2d(64, 128, 3, tf_padding_type="same"))
        self.add_module("relu21", nn.ReLU())

        self.add_module("conv22", _tfconv2d(128, 128, 3, tf_padding_type="same"))
        self.add_module("relu22", nn.ReLU())
        self.add_module("max_pool2", _tfmaxpool2d(2, stride=2, tf_padding_type="same"))

        self.add_module("conv31", _tfconv2d(128, 256, 3, tf_padding_type="same"))
        self.add_module("relu31", nn.ReLU())

        self.add_module("conv32", _tfconv2d(256, 256, 3, tf_padding_type="same"))
        self.add_module("relu32", nn.ReLU())

        self.add_module("conv33", _tfconv2d(256, 256, 3, tf_padding_type="same"))
        self.add_module("relu33", nn.ReLU())

        if variant == 19:
            self.add_module("conv34", _tfconv2d(256, 256, 3, tf_padding_type="same"))
            self.add_module("relu34", nn.ReLU())

        self.add_module("max_pool3", _tfmaxpool2d(2, stride=2, tf_padding_type="same"))

        self.add_module("conv41", _tfconv2d(256, 512, 3, tf_padding_type="same"))
        self.add_module("relu41", nn.ReLU())

        self.add_module("conv42", _tfconv2d(512, 512, 3, tf_padding_type="same"))
        self.add_module("relu42", nn.ReLU())

        self.add_module("conv43", _tfconv2d(512, 512, 3, tf_padding_type="same"))
        self.add_module("relu43", nn.ReLU())

        if variant == 19:
            self.add_module("conv44", _tfconv2d(512, 512, 3, tf_padding_type="same"))
            self.add_module("relu44", nn.ReLU())

        self.add_module("max_pool4", _tfmaxpool2d(2, stride=2, tf_padding_type="same"))

        self.add_module("conv51", _tfconv2d(512, 512, 3, tf_padding_type="same"))
        self.add_module("relu51", nn.ReLU())

        self.add_module("conv52", _tfconv2d(512, 512, 3, tf_padding_type="same"))
        self.add_module("relu52", nn.ReLU())

        self.add_module("conv53", _tfconv2d(512, 512, 3, tf_padding_type="same"))
        self.add_module("relu53", nn.ReLU())

        if variant == 19:
            self.add_module("conv54", _tfconv2d(512, 512, 3, tf_padding_type="same"))
            self.add_module("relu54", nn.ReLU())

        self.add_module("max_pool5", _tfmaxpool2d(2, stride=2, tf_padding_type="same"))

        self.add_module("flatten", nn.Flatten())

        self.add_module("dense1", nn.Linear(in_features=7 * 7 * 512, out_features=4096))
        self.add_module("relu1", nn.ReLU())
        self.add_module("dropout1", nn.Dropout(p=0.5))

        self.add_module("dense2", nn.Linear(in_features=4096, out_features=4096))
        self.add_module("relu2", nn.ReLU())
        self.add_module("dropout2", nn.Dropout(p=0.5))

        self.add_module("dense3", nn.Linear(in_features=4096, out_features=num_outputs))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)


class AllCNNC(nn.Sequential):
    """All Convolutional network C.

    Note: Proposed in

        - Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller
          Striving for Simplicity: The All Convolutional Net (2015).
    """

    def __init__(self):
        """Build the network."""
        super(AllCNNC, self).__init__()

        self.add_module("dropout1", nn.Dropout(p=0.2))

        self.add_module("conv1", _tfconv2d(3, 96, 3, tf_padding_type="same"))
        self.add_module("relu1", nn.ReLU())

        self.add_module("conv2", _tfconv2d(96, 96, 3, tf_padding_type="same"))
        self.add_module("relu2", nn.ReLU())

        self.add_module(
            "conv3",
            _tfconv2d(96, 96, 3, stride=(2, 2), tf_padding_type="same"),
        )
        self.add_module("relu3", nn.ReLU())

        self.add_module("dropout2", nn.Dropout(p=0.5))

        self.add_module("conv4", _tfconv2d(96, 192, 3, tf_padding_type="same"))
        self.add_module("relu4", nn.ReLU())

        self.add_module("conv5", _tfconv2d(192, 192, 3, tf_padding_type="same"))
        self.add_module("relu5", nn.ReLU())

        self.add_module(
            "conv6",
            _tfconv2d(192, 192, 3, stride=(2, 2), tf_padding_type="same"),
        )
        self.add_module("relu6", nn.ReLU())

        self.add_module("dropout3", nn.Dropout(p=0.5))

        self.add_module("conv7", _tfconv2d(192, 192, 3))
        self.add_module("relu7", nn.ReLU())

        self.add_module("conv8", _tfconv2d(192, 192, 1, tf_padding_type="same"))
        self.add_module("relu8", nn.ReLU())

        self.add_module("conv9", _tfconv2d(192, 100, 1, tf_padding_type="same"))
        self.add_module("relu9", nn.ReLU())

        self.add_module("mean", _mean_allcnnc())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.1)
                nn.init.xavier_normal_(module.weight)
