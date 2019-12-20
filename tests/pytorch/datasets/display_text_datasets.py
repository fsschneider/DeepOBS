# -*- coding: utf-8 -*-
"""Script to show text from DeepOBS text datasets."""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

import deepobs.pytorch.config as config
from deepobs.pytorch import datasets

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)



def display_text(dataset_cls, grid_size=5, phase="train"):
    """Display text from a DeepOBS text dataset.

  Args:
    dataset_cls: The DeepOBS dataset class to display text from. Is assumed to
        yield a tuple (x, y) of input and output text.
    phase (str): Images from this phase ('train', 'train_eval', 'test') will be
        displayed.
  """

    dataset = dataset_cls(batch_size=grid_size * grid_size)
    if phase == "train":
        iterator = iter(dataset._train_dataloader)
    elif phase == "train_eval":
        iterator = iter(dataset._train_eval_dataloader)
    elif phase == "test":
        iterator = iter(dataset._test_dataloader)
    else:
        raise ValueError("Choose 'phase' from ['train', 'train_eval', 'test'].")

    x_, y_ = next(iterator)
    x_next, y_next = next(iterator)  # Next batch, will be plotted in red
    label_dict = load_label_dict(dataset_cls.__name__)
    fig = plt.figure()
    for i in range(grid_size * grid_size):
        axis = fig.add_subplot(grid_size, grid_size, i + 1)
        input_txt = "".join([label_dict[char] for char in np.squeeze(x_[i])])
        output_txt = "".join([label_dict[char] for char in np.squeeze(y_[i])])
        # Next Batch, to check if text continues
        input_next_txt = "".join(
            [label_dict[char] for char in np.squeeze(x_next[i])]
        )
        output_next_txt = "".join(
            [label_dict[char] for char in np.squeeze(y_next[i])]
        )
        txt = (
            "*INPUT* \n"
            + input_txt
            + "\n \n *OUTPUT* \n"
            + output_txt
            + "\n \n \n *INPUT NEXT BATCH* \n"
            + input_next_txt
            + "\n \n *OUTPUT NEXT BATCH* \n"
            + output_next_txt
        )
        axis.text(0, 0, txt, fontsize=10)
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
    if dataset == "tolstoi":
        filepath = os.path.join(config.get_data_dir(), "tolstoi/vocab.pkl")
        label_dict = pickle.load(open(filepath, "rb"))
    else:
        label_dict = IdentityDict()
    return label_dict


class IdentityDict(dict):
    """An identity dictionary, return the key as value."""

    def __missing__(self, key):
        return key


if __name__ == "__main__":
    #    display_text(datasets.tolstoi, grid_size=2, phase="train")
    display_text(datasets.tolstoi, grid_size=2, phase="train_eval")
    #    display_text(datasets.tolstoi, grid_size=2, phase="test")

    plt.show()
