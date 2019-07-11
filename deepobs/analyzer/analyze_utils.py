import numpy as np
from matplotlib import cm
import os
from .shared_utils import _load_json


def _preprocess_reference_path(reference_path):
    os.chdir(reference_path)
    pathes = [path for path in os.listdir(reference_path) if os.path.isdir(path)]

    if 'num_epochs' in pathes[0]:    # path was a path to an optimizer
        return reference_path.split()
    else:    # path was a testproblem path
        return [os.path.join(reference_path, path) for path in pathes]


# TODO is compute speed up to date?
def compute_speed(setting_folder, conv_perf, metric):
    runs = [run for run in os.listdir(setting_folder) if run.endswith(".json")]
    # metrices
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for run in runs:
        json_data = _load_json(setting_folder, run)
        train_losses.append(json_data['train_losses'])
        test_losses.append(json_data['test_losses'])
        # just add accuracies to the aggregate if they are available
        if 'train_accuracies' in json_data:
            train_accuracies.append(json_data['train_accuracies'])
            test_accuracies.append(json_data['test_accuracies'])

    perf = np.array(eval(metric))
    if metric == "test_losses" or metric == "train_losses":
        # average over first time they reach conv perf (use num_epochs if conv perf is not reached)
        speed = np.mean(
            np.argmax(perf <= conv_perf, axis=1) +
            np.invert(np.max(perf <= conv_perf, axis=1)) *
            perf.shape[1])
    elif metric == "test_accuracies" or metric == "train_accuracies":
        speed = np.mean(
            np.argmax(perf >= conv_perf, axis=1) +
            np.invert(np.max(perf >= conv_perf, axis=1)) *
            perf.shape[1])
    else:
        raise NotImplementedError

    return speed


def make_legend_and_colors_consistent(axes):
    handles_and_labels = []
    for ax_col in range(len(axes[0,:])): # for each testproblem get the color coding of the optimizer
        handles_and_labels_tupel = axes[0, ax_col].get_legend_handles_labels()
        handles_and_labels.append(handles_and_labels_tupel)

    # at first get a unique list of all optimizers included in the figure and their color
    optimizers = []
    for handles, labels in handles_and_labels:
        for idx, label in enumerate(labels):
            if label not in optimizers:
                optimizers.append(label)

    # now get unique color for each optimizer
    colormap = cm.Dark2(np.linspace(0, 1, len(optimizers)))
    for color_idx, optimizer in enumerate(optimizers):
        for handles, labels in handles_and_labels:
            for idx, label in enumerate(labels):
                if optimizer == label:
                    handles[idx].set_color(colormap[color_idx])


def rescale_ax(ax):
    """Rescale an axis to include the most important data.

    Args:
        ax (matplotlib.axis): Handle to a matplotlib axis.

    """
    lines = ax.lines
    y_data = np.array([])
    y_limits = []
    for line in lines:
        if line.get_label() != "convergence_performance":
            y_data = np.append(y_data, line.get_ydata())
        else:
            y_limits.append(line.get_ydata()[0])
    if len(y_data)!=0:
        y_limits.append(np.percentile(y_data, 20))
        y_limits.append(np.percentile(y_data, 80))
        y_limits = [y_limits[0] * 0.9, y_limits[1] * 1.1]
        if y_limits[0] != y_limits[1]:
            ax.set_ylim([max(1e-10, y_limits[0]), y_limits[1]])
        ax.margins(x=0)
    else:
        ax.set_ylim([1.0, 2.0])
    return ax
