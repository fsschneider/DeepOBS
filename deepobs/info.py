"""Info file for DeepOBS."""

SMALL_TEST_SET = ["quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"]
LARGE_TEST_SET = ["fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"]


DATA_SET_NAMING = {
    "two": "2D",
    "quadratic": "Quadratic",
    "mnist": "MNIST",
    "fmnist": "F-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "svhn": "SVHN",
    "imagenet": "ImageNet",
    "tolstoi": "Tolstoi",
}
TP_NAMING = {
    "d_beale": "Beale",
    "d_branin": "Branin",
    "d_rosenbrock": "Rosenbrock",
    "deep": "Deep",
    "logreg": "Log. Reg.",
    "mlp": "MLP",
    "2c2d": "2c2d",
    "3c3d": "3c3d",
    "vae": "VAE",
    "vgg_16": "VGG 16",
    "vgg_19": "VGG 19",
    "allcnnc": "All-CNN-C",
    "wrn164": "Wide ResNet 16-4",
    "wrn404": "Wide ResNet 40-4",
    "inception_v3": "Inception-v3",
    "char_rnn": "Char RNN",
}

DEFAULT_TEST_PROBLEMS_SETTINGS = {
    "quadratic_deep": {"batch_size": 128, "num_epochs": 100},
    "mnist_vae": {"batch_size": 64, "num_epochs": 50},
    "fmnist_2c2d": {"batch_size": 128, "num_epochs": 100},
    "cifar10_3c3d": {"batch_size": 128, "num_epochs": 100},
    "fmnist_vae": {"batch_size": 64, "num_epochs": 100},
    "cifar100_allcnnc": {"batch_size": 256, "num_epochs": 350},
    "cifar100_wrn164": {"batch_size": 128, "num_epochs": 160},
    "cifar100_wrn404": {"batch_size": 128, "num_epochs": 160},
    "svhn_3c3d": {"batch_size": 128, "num_epochs": 100},
    "svhn_wrn164": {"batch_size": 128, "num_epochs": 160},
    "tolstoi_char_rnn": {"batch_size": 50, "num_epochs": 200},
    "mnist_2c2d": {"batch_size": 128, "num_epochs": 100},
    "mnist_mlp": {"batch_size": 128, "num_epochs": 100},
    "fmnist_mlp": {"batch_size": 128, "num_epochs": 100},
    "mnist_logreg": {"batch_size": 128, "num_epochs": 50},
    "fmnist_logreg": {"batch_size": 128, "num_epochs": 50},
}
