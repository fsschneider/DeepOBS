# -*- coding: utf-8 -*-
"""Script to visualize images from DeepOBS datasets."""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepobs.tensorflow import datasets
import deepobs.tensorflow.config as config


def denormalize_image(img):
    """Convert a normalized (float) image back to unsigned 8-bit images."""
    img -= np.min(img)
    img /= np.max(img)
    img *= 255.0
    return np.round(img).astype(np.uint8)


def display_images(dataset_cls, grid_size=5, phase="train"):
    """Display images from a DeepOBS data set.

  Args:
    dataset_cls: The DeepOBS dataset class to display images from. Is assumed to
        yield a tuple (x, y) of images and one-hot label vectors.
    grid_size (int): Will display grid_size**2 number of images.
    phase (str): Images from this phase ('train', 'train_eval', 'test') will be
        displayed.
  """
    tf.reset_default_graph()
    dataset = dataset_cls(batch_size=grid_size * grid_size)
    x, y = dataset.batch
    if phase == "train":
        init_op = dataset.train_init_op
    elif phase == "train_eval":
        init_op = dataset.train_eval_init_op
    elif phase == "test":
        init_op = dataset.test_init_op
    else:
        raise ValueError(
            "Choose 'phase' from ['train', 'train_eval', 'test'].")
    with tf.Session() as sess:
        sess.run(init_op)
        x_, y_ = sess.run([x, y])
    label_dict = load_label_dict(dataset_cls.__name__)
    fig = plt.figure()
    for i in range(grid_size * grid_size):
        axis = fig.add_subplot(grid_size, grid_size, i + 1)
        img = np.squeeze(denormalize_image(x_[i]))
        axis.imshow(img)
        # axis.set_title("Label {0:d}".format(np.argmax(y_[i])))
        axis.set_title(label_dict[np.argmax(y_[i])])
        axis.axis("off")
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.suptitle(dataset_cls.__name__ + " (" + phase + ")")
    fig.show()


def load_label_dict(dataset):
    """Get dict that translates from label number to humanly-readable class
    (e.g. from 1 -> automobile on cifar 10)

    Args:
        dataset (str): Name of the dataset.

    Returns:
        dict: Dictionary that translates from class number to class label.

    """
    if dataset == "cifar10":
        with open(
                os.path.join(config.get_data_dir(),
                             "cifar-10/batches.meta.txt")) as lookup_file:
            label_dict = lookup_file.readlines()
    elif dataset == "cifar100":
        with open(
                os.path.join(config.get_data_dir(),
                             "cifar-100/fine_label_names.txt")) as lookup_file:
            label_dict = lookup_file.readlines()
    elif dataset == "fmnist":
        label_dict = dict([(0, "T-shirt"), (1, "Trouser"), (2, "Pullover"),
                           (3, "Dress"), (4, "Coat"), (5, "Sandal"),
                           (6, "Shirt"), (7, "Sneaker"), (8, "Bag"),
                           (9, "Ankle boot")])
    elif dataset == "imagenet":
        label_file = os.path.join(
            os.path.realpath(
                os.path.join(os.getcwd(), os.path.dirname(__file__))),
            "imagenet_labels.txt")
        # Read from text file
        label_dict = {}
        i = 0
        with open(label_file) as f:
            for line in f:
                label_dict[i] = line.rstrip()
                i += 1
    else:
        label_dict = IdentityDict()
    return label_dict


class IdentityDict(dict):
    """An identity dictionary, return the key as value."""

    def __missing__(self, key):
        return key


if __name__ == "__main__":
    display_images(datasets.mnist, grid_size=5, phase="train")
    display_images(datasets.mnist, grid_size=5, phase="train_eval")
    display_images(datasets.mnist, grid_size=5, phase="test")

    display_images(datasets.fmnist, grid_size=5, phase="train")
    display_images(datasets.fmnist, grid_size=5, phase="train_eval")
    display_images(datasets.fmnist, grid_size=5, phase="test")

    display_images(datasets.cifar10, grid_size=5, phase="train")
    display_images(datasets.cifar10, grid_size=5, phase="train_eval")
    display_images(datasets.cifar10, grid_size=5, phase="test")

    display_images(datasets.cifar100, grid_size=5, phase="train")
    display_images(datasets.cifar100, grid_size=5, phase="train_eval")
    display_images(datasets.cifar100, grid_size=5, phase="test")

    display_images(datasets.svhn, grid_size=5, phase="train")
    display_images(datasets.svhn, grid_size=5, phase="train_eval")
    display_images(datasets.svhn, grid_size=5, phase="test")

    display_images(datasets.imagenet, grid_size=5, phase="train")
    display_images(datasets.imagenet, grid_size=5, phase="train_eval")
    display_images(datasets.imagenet, grid_size=5, phase="test")

    plt.show()
