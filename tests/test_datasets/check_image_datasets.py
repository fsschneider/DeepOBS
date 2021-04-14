"""Check to visually inspect the DeepOBS image data sets."""

import random

import matplotlib.pyplot as plt
import numpy as np

import deepobs.datasets.pytorch as pytorch_datasets
import deepobs.datasets.tensorflow as tensorflow_datasets
from tests.test_datasets import utils_datasets

# Basic Settings of the Test
BATCH_SIZE = 4
MODES = ["train", "train_eval", "valid", "test"]
GRID_SIZE = 2
# DEVICES = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
FRAMEWORKS = [
    "pytorch",
    # "tensorflow",
]

# Collect all possible test scenarios (devices, frameworks, etc.)
SCENARIOS = []
SCENARIO_IDS = []
for fw in FRAMEWORKS:
    datasets = globals()[fw + "_datasets"].__all__
    # Only show data sets that include images!
    SCENARIOS.extend(
        [
            (fw, ds)
            for ds in datasets
            if "image" in getattr(utils_datasets, ds.upper())["type"]
        ]
    )
    SCENARIO_IDS.extend([fw + ":" + ds for ds in datasets])


def display_images(framework, dataset):
    """Display images from the given DeepOBS data set.

    Args:
        framework (str): String of the framework.
        dataset (str): String of the data set to visualize.
    """
    data = getattr(globals()[fw + "_datasets"], dataset)(BATCH_SIZE)
    dataset_info = getattr(utils_datasets, dataset.upper())

    # Create Figure harness
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(framework + " " + dataset)
    outer_grid = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25)

    for idx, m in enumerate(MODES):
        dataloader = getattr(data, "_" + m + "_dataloader")
        iterator = iter(dataloader)
        inputs, labels = next(iterator)

        ax = fig.add_subplot(outer_grid[idx])
        ax.set_title(m.replace("_", " ").title())
        inner_grid = outer_grid[idx].subgridspec(GRID_SIZE, GRID_SIZE)
        axs = inner_grid.subplots()
        ax.axis("off")

        n_image = 0
        for _, img_ax in np.ndenumerate(axs):
            # print(labels[n_image])
            # Show image (permute since matplotlib expects HWC)
            image = inputs[n_image].permute(1, 2, 0).numpy()
            if "classification" in dataset_info["type"]:
                label = dataset_info["labels"][labels[n_image].item()]
            elif "attributes" in dataset_info["type"]:
                attribute_indx = [i for i, x in enumerate(labels[n_image]) if x == 1]
                label = dataset_info["labels"][random.choice(attribute_indx)]
            img_ax.imshow(utils_datasets.denormalize_image(image))
            img_ax.set_title(label, y=-0.2)
            img_ax.axis("off")

            n_image += 1

    plt.show()


if __name__ == "__main__":
    for scenario in SCENARIOS:
        display_images(*scenario)

    plt.show()
