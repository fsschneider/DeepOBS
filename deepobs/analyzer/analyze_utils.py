import numpy as np
import os


def _preprocess_path(path):
    """The path can either be a path to a specific optimizer or to a whole testproblem.
    In the latter case the path should become a list of all optimizers for that testproblem
    Args:
        path(str): Path to the optimizer or to a whole testproblem
    Returns:
        A list of all optimizers.
        """
    path = os.path.abspath(path)
    pathes = sorted([_path for _path in os.listdir(path) if os.path.isdir(os.path.join(path, _path))])

    if 'num_epochs' in pathes[0]:    # path was a path to an optimizer
        return path.split()
    else:    # path was a testproblem path
        return sorted([os.path.join(path, _path) for _path in pathes])


def _rescale_ax(ax):
    lines = ax.lines
    y_data = np.array([])
    y_limits = []
    for line in lines:
        y_data = np.append(y_data, line.get_ydata())

    if len(y_data) != 0:
        y_limits.append(np.percentile(y_data, 20))
        y_limits.append(np.percentile(y_data, 80))
        y_limits = [y_limits[0] * 0.9, y_limits[1] * 1.1]
        if y_limits[0] != y_limits[1]:
            ax.set_ylim([max(1e-10, y_limits[0]), y_limits[1]])
        ax.margins(x=0)
    else:
        ax.set_ylim([1.0, 2.0])
    return ax

