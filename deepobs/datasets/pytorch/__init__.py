"""PyTorch data sets for DeepOBS."""

from deepobs.datasets.pytorch.afhq import AFHQ
from deepobs.datasets.pytorch.celeba import CelebA
from deepobs.datasets.pytorch.cifar import CIFAR10, CIFAR100
from deepobs.datasets.pytorch.fmnist import FMNIST
from deepobs.datasets.pytorch.mnist import MNIST
from deepobs.datasets.pytorch.quadratic import Quadratic
from deepobs.datasets.pytorch.svhn import SVHN
from deepobs.datasets.pytorch.two_d import TwoD

__all__ = [
    "AFHQ",
    "CelebA",
    "CIFAR10",
    "CIFAR100",
    "FMNIST",
    "MNIST",
    "Quadratic",
    "SVHN",
    "TwoD",
]
