"""Display the image datasets."""

import matplotlib.pyplot as plt

from utils.utils_tests import get_datasets  # noqa

GRID_SIZE = 3


def display_images(dataset, framework, grid_size):
    """Display images of a given dataset in a grid.

    Args:
        dataset ([type]): [description]
        framework ([type]): [description]
        grid_size ([type]): [description]
    """
    fig, axes = plt.subplots(2 * grid_size + 1, 2 * grid_size + 1)

    title = dataset.capitalize() + " using " + framework.capitalize()
    fig.suptitle(title)
    fig.show()


if __name__ == "__main__":
    for framework, datasets in get_datasets().items():
        for dataset in datasets:
            if dataset != "two_d" and dataset != "tolstoi" and dataset != "quadratic":
                display_images(dataset, framework, grid_size=GRID_SIZE)

    plt.show()
