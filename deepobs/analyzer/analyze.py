
from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
from .shared_utils import create_setting_analyzer_ranking, _determine_available_metric, _get_optimizer_name_and_testproblem_from_path
from ..tuner.tuner_utils import generate_tuning_summary
from .analyze_utils import _rescale_ax, _preprocess_path
import pandas as pd


def plot_results_table(results_path, mode='most', metric='valid_accuracies', conv_perf_file=None):
    # TODO results table plots the valid metric, which is now what we want.
    # TODO general solution: implement the valid metric only for the setting analyzer ranking
    
    """Summarizes the performance of the optimizer and prints it to a pandas data frame.

            Args:
                results_path (str): The path to the results directory.
                mode (str): The mode by which to decide the best setting.
                metric (str): The metric by which to decide the best setting.
                conv_perf_file (str): Path to the convergence performance file. It is used to calculate the speed of the optimizer. Defaults to ``None`` in which case the speed measure is N.A.

            Returns:
                pandas.DataFrame: A data frame that summarizes the results.
                """
    table_dic = {}
    testproblems = os.listdir(results_path)
    metric_keys = ['Hyperparameters', 'Performance', 'Speed', 'Training Parameters']
    for testproblem in testproblems:
        # init new subdict for testproblem
        for metric_key in metric_keys:
            table_dic[(testproblem, metric_key)] = {}

        testproblem_path = os.path.join(results_path, testproblem)
        optimizers = os.listdir(testproblem_path)
        for optimizer in optimizers:
            optimizer_path = os.path.join(testproblem_path, optimizer)
            optimizer_performance_dic = get_performance_dictionary(optimizer_path, mode, metric, conv_perf_file)

            # invert inner dics for multiindexing
            for metric_key in metric_keys:
                table_dic[(testproblem, metric_key)][optimizer] = optimizer_performance_dic[metric_key]

    # correct multiindexing
    table = pd.DataFrame.from_dict(table_dic, orient='index')
    print(table)
    return table


def plot_testset_performances(results_path, mode = 'most', metric = 'valid_accuracies', reference_path = None):
    """Plots all optimizer performances for all testproblems.

    Args:
        results_path (str): The path to the results folder.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        reference_path(str): Path to the reference results folder. For each available reference testproblem, all optimizers are plotted as reference.

    Returns:
        matplotlib.axes.Axes: The axes with the plots.

        """
    testproblems = [path for path in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, path))]
    if reference_path is not None:
        reference_path = os.path.join(reference_path)
        reference_testproblems = [path for path in os.listdir(results_path) if os.path.isdir(os.path.join(reference_path, path))]
    else:
        reference_testproblems = []
    n_testproblems = len(testproblems)
    __, ax = plt.subplots(4, n_testproblems, sharex='col')
    for idx, testproblem in enumerate(testproblems):
        testproblem_path = os.path.join(results_path, testproblem)
        ax[:, idx] = _plot_optimizer_performance(testproblem_path, ax[:, idx], mode, metric)
        if testproblem in reference_testproblems:
            reference_testproblem_path = os.path.join(reference_path, testproblem)
            ax[:, idx] = _plot_optimizer_performance(reference_testproblem_path, ax[:, idx], mode, metric)

    metrices = ['test_losses', 'train_losses', 'test_accuracies', 'train_accuracies']
    for idx, _metric in enumerate(metrices):
        # label y axes
        ax[idx, 0].set_ylabel(_metric)
        # rescale
        for idx2 in range(n_testproblems):
            ax[idx, idx2] = _rescale_ax(ax[idx, idx2])
            ax[3, idx2].set_xlabel('epochs')
    # show legend of optimizers
    ax[0, 0].legend()
    plt.tight_layout()
    plt.show()
    return ax


def plot_hyperparameter_sensitivity_2d(optimizer_path, hyperparams, mode='final', metric = 'valid_accuracies', xscale='linear', yscale = 'linear'):
    param1, param2 = hyperparams
    metric = _determine_available_metric(optimizer_path, metric)
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    optimizer_name, testproblem = _get_optimizer_name_and_testproblem_from_path(optimizer_path)

    param_values1 = np.array([d['params'][param1] for d in tuning_summary])
    param_values2 = np.array([d['params'][param2] for d in tuning_summary])

    target_means = np.array([d['target_mean'] for d in tuning_summary])
    target_stds = [d['target_std'] for d in tuning_summary]

    _, ax = plt.subplots()

    con = ax.tricontourf(param_values1, param_values2, target_means, cmap = 'CMRmap', levels=len(target_means))
    ax.scatter(param_values1, param_values2)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    cbar = plt.colorbar(con)
    cbar.set_label(metric)
    plt.show()
    return ax


def _plot_hyperparameter_sensitivity(optimizer_path, hyperparam, ax, mode='final', metric = 'valid_accuracies',
                                    plot_std=False):

    metric = _determine_available_metric(optimizer_path, metric)
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    optimizer_name, testproblem = _get_optimizer_name_and_testproblem_from_path(optimizer_path)

    # create array for plotting
    param_values = [d['params'][hyperparam] for d in tuning_summary]
    target_means = [d['target_mean'] for d in tuning_summary]
    target_stds = [d['target_std'] for d in tuning_summary]

    param_values, target_means, target_stds = (list(t) for t in
                                               zip(*sorted(zip(param_values, target_means, target_stds))))

    param_values = np.array(param_values)
    target_means = np.array(target_means)
    ax.plot(param_values, target_means, linewidth=3, label=optimizer_name)
    if plot_std:
        ranks = create_setting_analyzer_ranking(optimizer_path, mode, metric)
        for rank in ranks:
            values = rank.get_all_final_values(metric)
            param_value = rank.aggregate['optimizer_hyperparams'][hyperparam]
            for value in values:
                ax.scatter(param_value, value, marker='x', color='b')
            ax.plot((param_value, param_value), (min(values), max(values)), color='grey', linestyle='--')
    ax.set_title(testproblem, fontsize=20)
    return ax


def plot_hyperparameter_sensitivity(path, hyperparam, mode='final', metric = 'valid_accuracies',
                                    xscale='linear',
                                    plot_std=True,
                                    reference_path = None):

    """Plots the hyperparameter sensitivtiy of the optimizer.

    Args:
        path (str): The path to the optimizer to analyse. Or to a whole testproblem. In that case, all optimizer sensitivities are plotted.
        hyperparam (str): The name of the hyperparameter that should be analyzed.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        xscale (str): The scale for the parameter axes. Is passed to plt.xscale().
        plot_std (bool): Whether to plot markers for individual seed runs or not. If `False`, only the mean is plotted.
        reference_path (str): Path to the reference optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are taken as reference).

    Returns:
        matplotlib.axes.Axes: The figure and axes of the plot.
        """

    _, ax = plt.subplots()
    pathes = _preprocess_path(path)
    for optimizer_path in pathes:
        ax = _plot_hyperparameter_sensitivity(optimizer_path, hyperparam, ax, mode, metric, plot_std)
    if reference_path is not None:
        pathes = _preprocess_path(reference_path)
        for reference_optimizer_path in pathes:
            ax = _plot_hyperparameter_sensitivity(reference_optimizer_path, hyperparam, ax, mode, metric, plot_std)

    plt.xscale(xscale)
    plt.xlabel(hyperparam, fontsize=16)
    plt.ylabel(metric, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend()
    plt.show()
    return ax


def plot_final_metric_vs_tuning_rank(optimizer_path, metric='valid_accuracies'):
    metric = _determine_available_metric(optimizer_path, metric)
    ranks = create_setting_analyzer_ranking(optimizer_path, mode='final', metric=metric)
    means = []
    fig, ax = plt.subplots()
    for idx, rank in enumerate(ranks):
        means.append(rank.get_final_value(metric))
        values = rank.get_all_final_values(metric)
        for value in values:
            ax.scatter(idx, value, marker='x', color='b')
        ax.plot((idx, idx), (min(values), max(values)), color= 'grey', linestyle='--')
    ax.plot(range(len(ranks)), means)
    optimizer, testproblem = _get_optimizer_name_and_testproblem_from_path(optimizer_path)
    ax.set_title(optimizer + ' on ' + testproblem)
    ax.set_xlabel('tuning rank')
    ax.set_ylabel(metric)
    plt.show()
    return fig, ax


def get_performance_dictionary(optimizer_path, mode = 'most', metric = 'valid_accuracies', conv_perf_file = None):
    """Summarizes the performance of the optimizer.

    Args:
        optimizer_path (str): The path to the optimizer to analyse.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        conv_perf_file (str): Path to the convergence performance file. It is used to calculate the speed of the optimizer. Defaults to ``None`` in which case the speed measure is N.A.

    Returns:
        dict: A dictionary that holds the best setting and it's performance.
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


def _plot_optimizer_performance(path, ax = None, mode = 'most', metric = 'valid_accuracies'):
    """Plots the training curve of an optimizer.

    Args:
        path (str): Path to the optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are plotted).
        ax (matplotlib.axes.Axes): The axes to plot the trainig curves for all metrices. Must have 4 subaxes.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
    Returns:
        matplotlib.axes.Axes: The axes with the plots.

        """
    metrices = ['test_losses', 'train_losses', 'test_accuracies', 'train_accuracies']
    if ax is None:  # create default axis for all 4 metrices
        _, ax = plt.subplots(4, 1, sharex='col')

    pathes = _preprocess_path(path)
    for optimizer_path in pathes:
        setting_analyzer_ranking = create_setting_analyzer_ranking(optimizer_path, mode, metric)
        setting = setting_analyzer_ranking[0]

        optimizer_name = os.path.basename(optimizer_path)
        for idx, _metric in enumerate(metrices):
            if _metric in setting.aggregate:
                mean = setting.aggregate[_metric]['mean']
                std = setting.aggregate[_metric]['std']
                ax[idx].plot(mean, label=optimizer_name)
                ax[idx].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    _, testproblem = _get_optimizer_name_and_testproblem_from_path(optimizer_path)
    ax[0].set_title(testproblem, fontsize=18)
    return ax


def plot_optimizer_performance(path, ax = None, mode = 'most', metric = 'valid_accuracies', reference_path = None):
    """Plots the training curve of optimizers and addionally plots reference results from the ``reference_path``

    Args:
        path (str): Path to the optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are plotted).
        ax (matplotlib.axes.Axes): The axes to plot the trainig curves for all metrices. Must have 4 subaxes (one for each metric).
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        reference_path (str): Path to the reference optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are taken as reference).

    Returns:
        matplotlib.axes.Axes: The axes with the plots.

        """

    ax = _plot_optimizer_performance(path, ax, mode, metric)
    if reference_path is not None:
        ax = _plot_optimizer_performance(reference_path, ax, mode, metric)

    metrices = ['test_losses', 'train_losses', 'test_accuracies', 'train_accuracies']
    for idx, _metric in enumerate(metrices):
        # set y labels
        ax[idx].set_ylabel(_metric, fontsize = 14)
        # rescale plots
        # ax[idx] = _rescale_ax(ax[idx])
        ax[idx].tick_params(labelsize=12)

    # show optimizer legends
    ax[0].legend(fontsize=12)

    ax[3].set_xlabel('epochs', fontsize = 14)

    plt.show()
    return ax

