from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
from .shared_utils import create_setting_analyzer_ranking, _determine_available_metric
from ..tuner.tuner_utils import generate_tuning_summary
from .analyze_utils import rescale_ax, _preprocess_reference_path


# TODO
def plot_performance_table(results_path):
    pass


def plot_all_testproblems_performances(results_path, mode = 'final', metric = 'test_accuracies', reference_path = None):
    """Plots all optimizer performances for all testproblems.
    Args:
        results_path (str): The path to the results folder.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        reference_path(str): Path to the reference results folder. For each available reference testproblem, all optimizers are plotted as reference.
    Returns:
        ax (plt.axes): The axes with the plots.
        """
    testproblems = [path for path in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, path))]
    if reference_path is not None:
        reference_testproblems = [path for path in os.listdir(results_path) if os.path.isdir(os.path.join(reference_path, path))]
    else:
        reference_testproblems = []
    n_testproblems = len(testproblems)
    __, ax = plt.subplots(4, n_testproblems)
    for idx, testproblem in enumerate(testproblems):
        testproblem_path = os.path.join(results_path, testproblem)
        ax[:, idx] = _plot_all_optimizer_performances_for_testproblem(testproblem_path, ax[:, idx], mode, metric)
        if testproblem in reference_testproblems:
            reference_testproblem_path = os.path.join(reference_path, testproblem)
            ax[:, idx] = _plot_all_optimizer_performances_for_testproblem(reference_testproblem_path, ax[:, idx], mode, metric)
    plt.show()
    return ax


def _plot_all_optimizer_performances_for_testproblem(testproblem_path, ax = None, mode = 'final', metric = 'test_accuracies'):
    """Plots the performance of all optimizers in one testproblem folder.
    Args:
        testproblem_path (str): The path to the testproblem.
        ax (plt.axes instance that has 4 subaxes (one for each possible metric)): The axes to plot the trainig curves for all metrices.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
    Returns:
        ax (plt.axes): The axes with the plots.
        """
    optimizers = [path for path in os.listdir(testproblem_path) if os.path.isdir(os.path.join(testproblem_path, path))]
    for optimizer in optimizers:
        optimizer_path = os.path.join(testproblem_path, optimizer)
        ax = _plot_optimizer_performance(optimizer_path, ax, mode, metric)
    return ax


def plot_hyperparameter_sensitivity(optimizer_path, hyperparam, mode='final', metric = 'test_accuracies', xscale='linear'):
    """Plots the hyperparameter sensitivtiy of the optimizer.
    Args:
        optimizer_path (str): The path to the optimizer to analyse.
        hyperparam (str): The name of the hyperparameter that should be analyzed.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        xscale (str): The scale for the parameter axes. Is passed to plt.xscale().
    Returns:
        fig, ax: The figure and axes of the plot.
        """
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    optimizer_name = os.path.split(optimizer_path)[-1]
    testproblem = os.path.split(optimizer_path)[-2]

    # create array for plotting
    param_values = [d['params'][hyperparam] for d in tuning_summary]
    target_means = [d['target_mean'] for d in tuning_summary]
    target_stds = [d['target_std'] for d in tuning_summary]
    # sort the values synchronised for plotting
    param_values, target_means, target_stds = (list(t) for t in zip(*sorted(zip(param_values, target_means, target_stds))))

    fig, ax = plt.subplots()
    param_values = np.array(param_values)
    target_means = np.array(target_means)
    target_stds = np.array(target_stds)
    ax.plot(param_values, target_means)
    ax.fill_between(param_values, target_means - target_stds, target_means + target_stds, alpha=0.3)
    plt.xscale(xscale)
    ax.set_title(optimizer_name + ' on ' + testproblem)
    plt.show()
    return fig, ax


def get_performance_dictionary(optimizer_path, mode = 'final', metric = 'test_accuracies', conv_perf_file = None):
    """Summarizes the performance of the optimizer.
    Args:
        optimizer_path (str): The path to the optimizer to analyse.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        conv_perf_file (str): Path to the convergence performance file. It is used to calculate the speed of the optimizer. Defaults to ``None`` in which case the speed measure is N.A.
    Returns:
        perf_dict (dict): A dictionary that holds the best setting and it's perormance.
        """
    metric = _determine_available_metric(optimizer_path, metric)
    setting_analyzers_ranking = create_setting_analyzer_ranking(optimizer_path, mode, metric)
    sett = setting_analyzers_ranking[0]

    perf_dict = dict()
    if mode == 'final':
        perf_dict['Performance'] = sett.get_final_value(metric)
    elif mode == 'best':
        perf_dict['Performance'] = sett.get_best_value(metric)
    elif mode == 'most':
        # default performance for most is final value
        perf_dict['Performance'] = sett.get_final_value(metric)
    else:
        raise NotImplementedError

    if conv_perf_file is not None:
        perf_dict['Speed'] = sett.calculate_speed(conv_perf_file)
    else:
        perf_dict['Speed'] = 'N.A.'

    perf_dict['Hyperparameters'] = sett.aggregate['optimizer_hyperparams']
    perf_dict['Training Parameters'] = sett.aggregate['training_params']
    return perf_dict


def _plot_optimizer_performance(optimizer_path, ax = None, mode = 'final', metric = 'test_accuracies'):
    """Plots the training curve of an optimizer.
    Args:
        optimizer_path (str): The path to the optimizer to analyse.
        ax (plt.axes instance that has 4 subaxes (one for each possible metric)): The axes to plot the trainig curves for all metrices.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
    Returns:
        ax (plt.axes): The axes with the plots.
        """
    setting_analyzer_ranking = create_setting_analyzer_ranking(optimizer_path, mode, metric)
    setting = setting_analyzer_ranking[0]

    metrices = ['train_losses', 'train_accuracies', 'test_losses', 'test_accuracies']
    if ax is None:    # create default axis for all 4 metrices
        _, ax = plt.subplots(4, 1)
    optimizer_name = os.path.basename(optimizer_path)
    for idx, _metric in enumerate(metrices):
        if _metric in setting.aggregate:
            mean = setting.aggregate[_metric]['mean']
            std = setting.aggregate[_metric]['std']
            ax[idx].plot(mean, label=optimizer_name)
            ax[idx].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
            ax[idx].legend()
    return ax


def plot_optimizer_performance(optimizer_path, ax = None, mode = 'final', metric = 'test_accuracies', reference_path = None):
    """Plots the training curve of an optimizer and addionally plots reference results from the ``reference_path``
    Args:
        optimizer_path (str): The path to the optimizer to analyse.
        ax (plt.axes instance that has 4 subaxes (one for each possible metric)): The axes to plot the trainig curves for all metrices.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        reference_path(str): Path to the reference optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are taken as reference)
    Returns:
        ax (plt.axes): The axes with the plots.
        """
    ax = _plot_optimizer_performance(optimizer_path, ax, mode, metric)
    if reference_path is not None:
        reference_path = _preprocess_reference_path(reference_path)    # reference path can either be an optimizer_path or a testproblem path
        for reference_optimizer_path in reference_path:
            ax = _plot_optimizer_performance(reference_optimizer_path, ax, mode, metric)
    plt.show()
    return ax