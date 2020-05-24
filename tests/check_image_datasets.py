import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import deepobs.pytorch.datasets as torch_datasets
import deepobs.tensorflow.datasets as tf_datasets
from utils.utils_tests import get_datasets  # noqa

GRID_SIZE = 3


def display_images(dataset, framework, grid_size):
    fig, axes = plt.subplots(2 * grid_size + 1, 2 * grid_size + 1)

    title = dataset.capitalize() + " using " + framework.capitalize()
    fig.suptitle(title)
    fig.show()


if __name__ == "__main__":
    for framework, datasets in get_datasets().items():
        for dataset in datasets:
            if (
                dataset != "two_d"
                and dataset != "tolstoi"
                and dataset != "quadratic"
            ):
                display_images(dataset, framework, grid_size=GRID_SIZE)

    plt.show()
