"""PyTorch models for DeepOBS."""

from deepobs.models.pytorch.basic import MLP, LogReg, QuadraticDeep
from deepobs.models.pytorch.convnet import VGG, AllCNNC, Basic2c2d, Basic3c3d
from deepobs.models.pytorch.gan import DCGAN_D, DCGAN_G
from deepobs.models.pytorch.resnet import WRN
from deepobs.models.pytorch.vae import VAE

__all__ = [
    "QuadraticDeep",
    "LogReg",
    "MLP",
    "Basic2c2d",
    "Basic3c3d",
    "VGG",
    "AllCNNC",
    "WRN",
    "VAE",
    "DCGAN_G",
    "DCGAN_D",
]
