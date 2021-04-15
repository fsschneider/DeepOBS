"""Provides information about the DeepOBS models."""

# TODO Computation of parameters is a bit weird. Variants are currenty not included
# Instead have functions that return the list of parameters based on inputs.
# This removes the global variables.
# Then we can also test the models with different inputs

QUADRATICDEEP = {
    "parameters": [100],
}

LOGREG_DEFAULT_CLASSES = 10
LOGREG = {
    "parameters": [784 * LOGREG_DEFAULT_CLASSES, LOGREG_DEFAULT_CLASSES],
}

MLP_DEFAULT_CLASSES = 10
MLP = {
    "parameters": [
        28 * 28 * 1000,
        1000,
        1000 * 500,
        500,
        500 * 100,
        100,
        100 * MLP_DEFAULT_CLASSES,
        MLP_DEFAULT_CLASSES,
    ],
}

BASIC2C2D_DEFAULT_CLASSES = 10
BASIC2C2D = {
    "parameters": [
        1 * 32 * 5 * 5,
        32,
        32 * 64 * 5 * 5,
        64,
        7 * 7 * 64 * 1024,
        1024,
        1024 * BASIC2C2D_DEFAULT_CLASSES,
        BASIC2C2D_DEFAULT_CLASSES,
    ],
}

BASIC3C3D_DEFAULT_CLASSES = 10
BASIC3C3D = {
    "parameters": [
        3 * 64 * 5 * 5,
        64,
        64 * 96 * 3 * 3,
        96,
        96 * 128 * 3 * 3,
        128,
        3 * 3 * 128 * 512,
        512,
        512 * 256,
        256,
        256 * BASIC3C3D_DEFAULT_CLASSES,
        BASIC3C3D_DEFAULT_CLASSES,
    ],
}

VGG_DEFAULT_CLASSES = 100
VGG_DEFAULT_VARIANT = 16
VGG = {
    "parameters": [
        3 * 64 * 3 * 3,
        64,
        64 * 64 * 3 * 3,
        64,
        64 * 128 * 3 * 3,
        128,
        128 * 128 * 3 * 3,
        128,
        128 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 256 * 3 * 3,
        256,
        256 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 512 * 3 * 3,
        512,
        512 * 7 * 7 * 4096,
        4096,
        4096 * 4096,
        4096,
        4096 * VGG_DEFAULT_CLASSES,
        VGG_DEFAULT_CLASSES,
    ],
}

ALLCNNC = {
    "parameters": [
        3 * 96 * 3 * 3,
        96,
        96 * 96 * 3 * 3,
        96,
        96 * 96 * 3 * 3,
        96,
        96 * 192 * 3 * 3,
        192,
        192 * 192 * 3 * 3,
        192,
        192 * 192 * 3 * 3,
        192,
        192 * 192 * 3 * 3,
        192,
        192 * 192 * 1 * 1,
        192,
        192 * 100 * 1 * 1,
        100,
    ],
}

WRN_DEFAULT_CLASSES = 100
WRN_DEFAULT_NUM_BLOCKS = 2
WRN_DEFAULT_WIDENING_FACTOR = 4
WRN = {
    "parameters": [
        3 * 16 * 3 * 3,
        16,
        16,
        16 * 64 * 1 * 1,
        16 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 64 * 3 * 3,
        64,
        64,
        64 * 128 * 1 * 1,
        64 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 128 * 3 * 3,
        128,
        128,
        128 * 256 * 1 * 1,
        128 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * 256 * 3 * 3,
        256,
        256,
        256 * WRN_DEFAULT_CLASSES,
        WRN_DEFAULT_CLASSES,
    ],
}

VAE_DEFAULT_LATENT_SPACE = 10
VAE = {
    "parameters": [
        1 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        7 * 7 * 64 * VAE_DEFAULT_LATENT_SPACE,
        VAE_DEFAULT_LATENT_SPACE,
        7 * 7 * 64 * VAE_DEFAULT_LATENT_SPACE,
        VAE_DEFAULT_LATENT_SPACE,
        VAE_DEFAULT_LATENT_SPACE * 24,
        24,
        24 * 49,
        49,
        1 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        64 * 64 * 4 * 4,
        64,
        14 * 14 * 64 * 28 * 28,
        28 * 28,
    ],
}

DCGAN_G_DEFAULT_CHANNELS = 3
DCGAN_G_DEFAULT_FM_SIZE = 64
DCGAN_G_DEFAULT_LATENT_SPACE = 100
DCGAN_G = {
    "parameters": [
        DCGAN_G_DEFAULT_LATENT_SPACE * DCGAN_G_DEFAULT_FM_SIZE * 8 * 4 * 4,
        DCGAN_G_DEFAULT_FM_SIZE * 8,
        DCGAN_G_DEFAULT_FM_SIZE * 8,
        DCGAN_G_DEFAULT_FM_SIZE * 8 * DCGAN_G_DEFAULT_FM_SIZE * 4 * 4 * 4,
        DCGAN_G_DEFAULT_FM_SIZE * 4,
        DCGAN_G_DEFAULT_FM_SIZE * 4,
        DCGAN_G_DEFAULT_FM_SIZE * 4 * DCGAN_G_DEFAULT_FM_SIZE * 2 * 4 * 4,
        DCGAN_G_DEFAULT_FM_SIZE * 2,
        DCGAN_G_DEFAULT_FM_SIZE * 2,
        DCGAN_G_DEFAULT_FM_SIZE * 2 * DCGAN_G_DEFAULT_FM_SIZE * 4 * 4,
        DCGAN_G_DEFAULT_FM_SIZE,
        DCGAN_G_DEFAULT_FM_SIZE,
        DCGAN_G_DEFAULT_FM_SIZE * DCGAN_G_DEFAULT_CHANNELS * 4 * 4,
    ]
}

DCGAN_D_DEFAULT_CHANNELS = 3
DCGAN_D_DEFAULT_FM_SIZE = 64
DCGAN_D = {
    "parameters": [
        DCGAN_D_DEFAULT_CHANNELS * DCGAN_D_DEFAULT_FM_SIZE * 4 * 4,
        DCGAN_D_DEFAULT_FM_SIZE * DCGAN_D_DEFAULT_FM_SIZE * 2 * 4 * 4,
        DCGAN_D_DEFAULT_FM_SIZE * 2,
        DCGAN_D_DEFAULT_FM_SIZE * 2,
        DCGAN_D_DEFAULT_FM_SIZE * 2 * DCGAN_D_DEFAULT_FM_SIZE * 4 * 4 * 4,
        DCGAN_D_DEFAULT_FM_SIZE * 4,
        DCGAN_D_DEFAULT_FM_SIZE * 4,
        DCGAN_D_DEFAULT_FM_SIZE * 4 * DCGAN_D_DEFAULT_FM_SIZE * 8 * 4 * 4,
        DCGAN_D_DEFAULT_FM_SIZE * 8,
        DCGAN_D_DEFAULT_FM_SIZE * 8,
        DCGAN_D_DEFAULT_FM_SIZE * 8 * 4 * 4,
    ]
}
