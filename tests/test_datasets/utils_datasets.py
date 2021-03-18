"""Utility functions for testing the DeepOBS data sets."""

MNIST = {
    "type": "image_classification",
    "channels": 1,
    "height": 28,
    "width": 28,
    "classes": 10,
    "n_train": 50000,
    "n_test": 10000,
    "n_valid": 10000,
    "n_train_eval": 10000,
    "input_type": float,
    "label_shape": 1,
    "label_type": int,
}

FMNIST = MNIST

CIFAR10 = {
    "type": "image_classification",
    "channels": 3,
    "height": 32,
    "width": 32,
    "classes": 10,
    "n_train": 40000,
    "n_test": 10000,
    "n_valid": 10000,
    "n_train_eval": 10000,
    "input_type": float,
    "label_shape": 1,
    "label_type": int,
}

CIFAR100 = CIFAR10
CIFAR100["classes"] = 1000

SVHN = {
    "type": "image_classification",
    "channels": 3,
    "height": 32,
    "width": 32,
    "classes": 10,
    "n_train": 73257 - 26032,
    "n_test": 26032,
    "n_valid": 26032,
    "n_train_eval": 26032,
    "input_type": float,
    "label_shape": 1,
    "label_type": int,
}

CELEBA = {
    "type": "image_attributes",
    "channels": 3,
    "height": 218,
    "width": 178,
    "classes": 10,
    "n_train": 162770,
    "n_test": 19962,
    "n_valid": 19867,
    "n_train_eval": 19962,
    "input_type": float,
    "label_shape": 2,
    "label_length": 40,
    "label_type": int,
}

AFHQ = {
    "type": "image_classification",
    "channels": 3,
    "height": 512,
    "width": 512,
    "classes": 3,
    "n_train": 13130,
    "n_test": 1500,
    "n_valid": 1500,
    "n_train_eval": 1500,
    "input_type": float,
    "label_shape": 1,
    "label_type": int,
}

QUADRATIC = {
    "type": "toy_problem",
    "dimension": 100,
    "input_shape": 2,
    "n_train": 1000,
    "n_test": 1000,
    "n_valid": 1000,
    "n_train_eval": 1000,
    "input_type": float,
    "label_shape": 2,
    "label_type": float,
}

TWOD = {
    "type": "toy_problem",
    "dimension": 100,
    "input_shape": 1,
    "n_train": 10000,
    "n_test": 10000,
    "n_valid": 10000,
    "n_train_eval": 10000,
    "input_type": float,
    "label_shape": 1,
    "label_type": float,
}
