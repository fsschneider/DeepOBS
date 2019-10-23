from __future__ import print_function

import os
import time
from collections import Counter

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from ..tuner.tuner_utils import generate_tuning_summary
from .analyze_utils import _preprocess_path, _rescale_ax
from .shared_utils import (
    _check_output_structure,
    _check_setting_folder_is_not_empty,
    _determine_available_metric,
    _get_optimizer_name_and_testproblem_from_path,
    create_setting_analyzer_ranking,
)

sns.set()
sns.set_style(
    "whitegrid",
    {
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": ".5",
        "ytick.color": ".5",
    },
)


def check_output(results_path):
    """Iterates through the results folder an checks all outputs for format and completeness. It checks for some basic
    format in every json file and looks for setting folders which are empty.
    It further gives an overview over the amount of different settings and seed runs for
    each test problem and each optimizer. It does not return anything, but it prints an overview to the console.

    Args:
        results_path (str): Path to the results folder.
    """
    testproblems = sorted(os.listdir(results_path))
    for testproblem in testproblems:
        testproblem_path = os.path.join(results_path, testproblem)
        optimizers = sorted(os.listdir(testproblem_path))
        for optimizer in optimizers:
            optimizer_path = os.path.join(testproblem_path, optimizer)
            settings = [
                setting
                for setting in os.listdir(optimizer_path)
                if os.path.isdir(os.path.join(optimizer_path, setting))
                and "num_epochs" in setting
            ]
            n_runs_list = []
            for setting in settings:
                setting_path = os.path.join(optimizer_path, setting)
                _check_setting_folder_is_not_empty(setting_path)
                jsons_files = [
                    file for file in os.listdir(setting_path) if "json" in file
                ]
                n_runs_list.append(len(jsons_files))
                for json_file in jsons_files:
                    _check_output_structure(setting_path, json_file)
            counter = Counter(n_runs_list)
            for n_runs, count in counter.items():
                print(
                    "{0:s} | {1:s}: {2:d} setting(s) with {3:d} seed(s).".format(
                        testproblem, optimizer, count, n_runs
                    )
                )


def estimate_runtime(
    framework,
    runner_cls,
    optimizer_cls,
    optimizer_hp,
    optimizer_hyperparams,
    n_runs=5,
    sgd_lr=0.01,
    testproblem="mnist_mlp",
    num_epochs=5,
    batch_size=128,
    **kwargs
):
    """Can be used to estimates the runtime overhead of a new optimizer compared to SGD. Runs the new optimizer and
    SGD seperately and calculates the fraction of wall clock overhead.

    Args:
        framework (str): Framework that you use. Must be 'pytorch' or 'tensorlfow'.
        runner_cls: The runner class that your optimizer uses.
        optimizer_cls: Your optimizer class.
        optimizer_hp (dict): Its hyperparameter specification as it is used in the runner initialization.
        optimizer_hyperparams (dict): Optimizer hyperparameter values to run.
        n_runs (int): The number of run calls for which the overhead is averaged over.
        sgd_lr (float): The vanilla SGD learning rate to use.
        testproblem (str): The deepobs testproblem to run SGD and the new optimizer on.
        num_epochs (int): The number of epochs to run for the testproblem.
        batch_size (int): Batch size of the testproblem.

    Returns:
        str: The output that is printed to the console.
    """

    # get the standard runner with SGD
    if framework == "pytorch":
        from deepobs import pytorch as ptobs
        from torch.optim import SGD

        runner_sgd = ptobs.runners.StandardRunner
        optimizer_class_sgd = SGD
        hp_sgd = {"lr": {"type": float}}
        hyperparams_sgd = {"lr": sgd_lr}

    elif framework == "tensorflow":
        from deepobs import tensorflow as tfobs
        import tensorflow as tf

        optimizer_class_sgd = tf.train.GradientDescentOptimizer
        hp_sgd = {"learning_rate": {"type": float}}
        runner_sgd = tfobs.runners.StandardRunner
        hyperparams_sgd = {"learning_rate": sgd_lr}
    else:
        raise RuntimeError("Framework must be pytorch or tensorflow")

    sgd_times = []
    new_opt_times = []

    for i in range(n_runs):
        print("** Start Run: ", i + 1, "of", n_runs)

        # SGD
        print("Running SGD")
        start_sgd = time.time()
        runner = runner_sgd(optimizer_class_sgd, hp_sgd)
        runner.run(
            testproblem=testproblem,
            hyperparams=hyperparams_sgd,
            batch_size=batch_size,
            num_epochs=num_epochs,
            no_logs=True,
            **kwargs
        )
        end_sgd = time.time()

        sgd_times.append(end_sgd - start_sgd)
        print("Time for SGD run ", i + 1, ": ", sgd_times[-1])

        # New Optimizer
        runner = runner_cls(optimizer_cls, optimizer_hp)
        print("Running...", optimizer_cls.__name__)
        start_script = time.time()
        runner.run(
            testproblem=testproblem,
            hyperparams=optimizer_hyperparams,
            batch_size=batch_size,
            num_epochs=num_epochs,
            no_logs=True,
            **kwargs
        )
        end_script = time.time()

        new_opt_times.append(end_script - start_script)
        print("Time for new optimizer run ", i + 1, ": ", new_opt_times[-1])

    overhead = np.divide(new_opt_times, sgd_times)

    output = (
        "** Mean run time SGD: "
        + str(np.mean(sgd_times))
        + "\n"
        + "** Mean run time new optimizer: "
        + str(np.mean(new_opt_times))
        + "\n"
        + "** Overhead per run: "
        + str(overhead)
        + "\n"
        + "** Mean overhead: "
        + str(np.mean(overhead))
        + " Standard deviation: "
        + str(np.std(overhead))
    )

    print(output)
    return output


def plot_results_table(
    results_path, mode="most", metric="valid_accuracies", conv_perf_file=None
):
    """Summarizes the performance of the optimizer and prints it to a pandas data frame.

            Args:
                results_path (str): The path to the results directory.
                mode (str): The mode by which to decide the best setting.
                metric (str): The metric by which to decide the best setting.
                conv_perf_file (str): Path to the convergence performance file. It is used to calculate the speed of the optimizer. Defaults to ``None`` in which case the speed measure is N.A.

            Returns:
                pandas.DataFrame: A data frame that summarizes the results on the test set.
                """
    table_dic = {}
    testproblems = os.listdir(results_path)
    metric_keys = [
        "Hyperparameters",
        "Performance",
        "Speed",
        "Training Parameters",
    ]
    for testproblem in testproblems:
        # init new subdict for testproblem
        for metric_key in metric_keys:
            table_dic[(testproblem, metric_key)] = {}

        testproblem_path = os.path.join(results_path, testproblem)
        optimizers = sorted(os.listdir(testproblem_path))
        for optimizer in optimizers:
            optimizer_path = os.path.join(testproblem_path, optimizer)
            optimizer_performance_dic = get_performance_dictionary(
                optimizer_path, mode, metric, conv_perf_file
            )

            # invert inner dics for multiindexing
            for metric_key in metric_keys:
                table_dic[(testproblem, metric_key)][
                    optimizer
                ] = optimizer_performance_dic[metric_key]

    # correct multiindexing
    table = pd.DataFrame.from_dict(table_dic, orient="index")
    print(table)
    return table


def plot_testset_performances(
    results_path,
    mode="most",
    metric="valid_accuracies",
    reference_path=None,
    show=True,
    which="mean_and_std",
):
    """Plots all optimizer performances for all testproblems.

    Args:
        results_path (str): The path to the results folder.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        reference_path (str): Path to the reference results folder. For each available reference testproblem, all optimizers are plotted as reference.
        show (bool): Whether to show the plot or not.
        which (str): ['mean_and_std', 'median_and_quartiles'] Solid plot mean or median, shaded plots standard deviation or lower/upper quartiles.

    Returns:
        tuple: The figure and axes.
        """
    testproblems = sorted(
        [
            path
            for path in os.listdir(results_path)
            if os.path.isdir(os.path.join(results_path, path))
        ]
    )
    if reference_path is not None:
        reference_path = os.path.join(reference_path)
        reference_testproblems = sorted(
            [
                path
                for path in os.listdir(results_path)
                if os.path.isdir(os.path.join(reference_path, path))
            ]
        )
    else:
        reference_testproblems = []
    n_testproblems = len(testproblems)
    fig, ax = plt.subplots(4, n_testproblems, sharex="col")
    for idx, testproblem in enumerate(testproblems):
        testproblem_path = os.path.join(results_path, testproblem)
        fig, ax[:, idx] = _plot_optimizer_performance(
            testproblem_path,
            fig=fig,
            ax=ax[:, idx],
            mode=mode,
            metric=metric,
            which=which,
        )
        if testproblem in reference_testproblems:
            reference_testproblem_path = os.path.join(
                reference_path, testproblem
            )
            fig, ax[:, idx] = _plot_optimizer_performance(
                reference_testproblem_path,
                fig=fig,
                ax=ax[:, idx],
                mode=mode,
                metric=metric,
                which=which,
            )

    metrices = ["Test Loss", "Train Loss", "Test Accuracy", "Train Accuracy"]
    for idx, _metric in enumerate(metrices):
        # label y axes
        ax[idx, 0].set_ylabel(_metric)
        # rescale
        for idx2 in range(n_testproblems):
            ax[idx, idx2] = _rescale_ax(ax[idx, idx2])
            ax[idx, idx2].xaxis.set_ticks_position("none")
            ax[3, idx2].set_xlabel("Epochs")
            ax[3, idx2].xaxis.set_ticks_position("bottom")
    # show legend of optimizers
    ax[0, 0].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.15)
    if show:
        plt.show()
    return fig, ax


def plot_hyperparameter_sensitivity_2d(
    optimizer_path,
    hyperparams,
    mode="final",
    metric="valid_accuracies",
    xscale="linear",
    yscale="linear",
    show=True,
):
    param1, param2 = hyperparams
    metric = _determine_available_metric(optimizer_path, metric)
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    optimizer_name, testproblem = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )

    param_values1 = np.array([d["params"][param1] for d in tuning_summary])
    param_values2 = np.array([d["params"][param2] for d in tuning_summary])

    target_means = np.array([d[metric + "_mean"] for d in tuning_summary])
    target_stds = [d[metric + "_std"] for d in tuning_summary]

    fig, ax = plt.subplots()

    con = ax.tricontourf(
        param_values1,
        param_values2,
        target_means,
        cmap="CMRmap",
        levels=len(target_means),
    )
    ax.scatter(param_values1, param_values2)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    cbar = plt.colorbar(con)
    cbar.set_label(metric)
    if show:
        plt.show()
    return fig, ax


def _plot_hyperparameter_sensitivity(
    optimizer_path,
    hyperparam,
    ax,
    mode="final",
    metric="valid_accuracies",
    plot_std=False,
):

    metric = _determine_available_metric(optimizer_path, metric)
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    optimizer_name, testproblem = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )

    # create array for plotting
    param_values = [d["params"][hyperparam] for d in tuning_summary]
    target_means = [d[metric + "_mean"] for d in tuning_summary]
    target_stds = [d[metric + "_mean"] for d in tuning_summary]

    param_values, target_means, target_stds = (
        list(t)
        for t in zip(*sorted(zip(param_values, target_means, target_stds)))
    )

    param_values = np.array(param_values)
    target_means = np.array(target_means)
    ax.plot(param_values, target_means, linewidth=3, label=optimizer_name)
    if plot_std:
        ranks = create_setting_analyzer_ranking(optimizer_path, mode, metric)
        for rank in ranks:
            values = rank.get_all_final_values(metric)
            param_value = rank.aggregate["optimizer_hyperparams"][hyperparam]
            for value in values:
                ax.scatter(param_value, value, marker="x", color="b")
            ax.plot(
                (param_value, param_value),
                (min(values), max(values)),
                color="grey",
                linestyle="--",
            )
    ax.set_title(testproblem, fontsize=20)
    return ax


def plot_hyperparameter_sensitivity(
    path,
    hyperparam,
    mode="final",
    metric="valid_accuracies",
    xscale="linear",
    plot_std=True,
    reference_path=None,
    show=True,
):
    """Plots the hyperparameter sensitivtiy of the optimizer.

    Args:
        path (str): The path to the optimizer to analyse. Or to a whole testproblem. In that case, all optimizer sensitivities are plotted.
        hyperparam (str): The name of the hyperparameter that should be analyzed.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        xscale (str): The scale for the parameter axes. Is passed to plt.xscale().
        plot_std (bool): Whether to plot markers for individual seed runs or not. If `False`, only the mean is plotted.
        reference_path (str): Path to the reference optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are taken as reference).
        show (bool): Whether to show the plot or not.

    Returns:
        tuple: The figure and axes of the plot.
        """
    fig, ax = plt.subplots()
    pathes = _preprocess_path(path)
    for optimizer_path in pathes:
        metric = _determine_available_metric(optimizer_path, metric)
        ax = _plot_hyperparameter_sensitivity(
            optimizer_path, hyperparam, ax, mode, metric, plot_std
        )
    if reference_path is not None:
        pathes = _preprocess_path(reference_path)
        for reference_optimizer_path in pathes:
            metric = _determine_available_metric(
                reference_optimizer_path, metric
            )
            ax = _plot_hyperparameter_sensitivity(
                reference_optimizer_path, hyperparam, ax, mode, metric, plot_std
            )

    plt.xscale(xscale)
    plt.xlabel(hyperparam, fontsize=16)
    plt.ylabel(metric, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend()
    if show:
        plt.show()
    return fig, ax


def plot_final_metric_vs_tuning_rank(
    optimizer_path, metric="valid_accuracies", show=True
):
    metric = _determine_available_metric(optimizer_path, metric)
    ranks = create_setting_analyzer_ranking(
        optimizer_path, mode="final", metric=metric
    )
    means = []
    fig, ax = plt.subplots()
    for idx, rank in enumerate(ranks):
        means.append(rank.get_final_value(metric))
        values = rank.get_all_final_values(metric)
        for value in values:
            ax.scatter(idx, value, marker="x", color="b")
        ax.plot(
            (idx, idx), (min(values), max(values)), color="grey", linestyle="--"
        )
    ax.plot(range(len(ranks)), means)
    optimizer, testproblem = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )
    ax.set_title(optimizer + " on " + testproblem)
    ax.set_xlabel("tuning rank")
    ax.set_ylabel(metric)
    if show:
        plt.show()
    return fig, ax


def get_performance_dictionary(
    optimizer_path, mode="most", metric="valid_accuracies", conv_perf_file=None
):
    """Summarizes the performance of the optimizer.

    Args:
        optimizer_path (str): The path to the optimizer to analyse.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        conv_perf_file (str): Path to the convergence performance file. It is used to calculate the speed of the optimizer. Defaults to ``None`` in which case the speed measure is N.A.

    Returns:
        dict: A dictionary that holds the best setting and it's performance on the test set.
        """
    metric = _determine_available_metric(optimizer_path, metric)
    setting_analyzers_ranking = create_setting_analyzer_ranking(
        optimizer_path, mode, metric
    )
    sett = setting_analyzers_ranking[0]

    perf_dict = dict()
    metric = (
        "test_accuracies"
        if "test_accuracies" in sett.aggregate
        else "test_losses"
    )
    if mode == "final":
        perf_dict["Performance"] = sett.get_final_value(metric)
    elif mode == "best":
        perf_dict["Performance"] = sett.get_best_value(metric)
    elif mode == "most":
        # default performance for most is final value
        perf_dict["Performance"] = sett.get_final_value(metric)
    else:
        raise NotImplementedError

    if conv_perf_file is not None:
        perf_dict["Speed"] = sett.calculate_speed(conv_perf_file)
    else:
        perf_dict["Speed"] = "N.A."

    perf_dict["Hyperparameters"] = sett.aggregate["optimizer_hyperparams"]
    perf_dict["Training Parameters"] = sett.aggregate["training_params"]
    return perf_dict


def _plot_optimizer_performance(
    path,
    fig=None,
    ax=None,
    mode="most",
    metric="valid_accuracies",
    which="mean_and_std",
):
    """Plots the training curve of an optimizer.

    Args:
        path (str): Path to the optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are plotted).
        fig (matplotlib.Figure): Figure to plot the training curves in.
        ax (matplotlib.axes.Axes): The axes to plot the trainig curves for all metrices. Must have 4 subaxes.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        which (str): ['mean_and_std', 'median_and_quartiles'] Solid plot mean or median, shaded plots standard deviation or lower/upper quartiles.

    Returns:
        matplotlib.axes.Axes: The axes with the plots.
        """
    metrices = [
        "test_losses",
        "train_losses",
        "test_accuracies",
        "train_accuracies",
    ]
    if ax is None:  # create default axis for all 4 metrices
        fig, ax = plt.subplots(4, 1, sharex="col")

    pathes = _preprocess_path(path)
    for optimizer_path in pathes:
        setting_analyzer_ranking = create_setting_analyzer_ranking(
            optimizer_path, mode, metric
        )
        setting = setting_analyzer_ranking[0]

        optimizer_name = os.path.basename(optimizer_path)
        for idx, _metric in enumerate(metrices):
            if _metric in setting.aggregate:

                if which == "mean_and_std":
                    center = setting.aggregate[_metric]["mean"]
                    std = setting.aggregate[_metric]["std"]
                    low, high = center - std, center + std
                elif which == "median_and_quartiles":
                    center = setting.aggregate[_metric]["median"]
                    low = setting.aggregate[_metric]["lower_quartile"]
                    high = setting.aggregate[_metric]["upper_quartile"]
                else:
                    raise ValueError("Unknown value which={}".format(which))

                ax[idx].plot(center, label=optimizer_name)
                ax[idx].fill_between(range(len(center)), low, high, alpha=0.3)

    _, testproblem = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path
    )
    ax[0].set_title(testproblem, fontsize=18)
    return fig, ax


def plot_optimizer_performance(
    path,
    fig=None,
    ax=None,
    mode="most",
    metric="valid_accuracies",
    reference_path=None,
    show=True,
    which="mean_and_std",
):
    """Plots the training curve of optimizers and addionally plots reference results from the ``reference_path``

    Args:
        path (str): Path to the optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are plotted).
        fig (matplotlib.Figure): Figure to plot the training curves in.
        ax (matplotlib.axes.Axes): The axes to plot the trainig curves for all metrices. Must have 4 subaxes (one for each metric).
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
        reference_path (str): Path to the reference optimizer or to a whole testproblem (in this case all optimizers in the testproblem folder are taken as reference).
        show (bool): Whether to show the plot or not.
        which (str): ['mean_and_std', 'median_and_quartiles'] Solid plot mean or median, shaded plots standard deviation or lower/upper quartiles.

    Returns:
        tuple: The figure and axes with the plots.

        """

    fig, ax = _plot_optimizer_performance(
        path, fig, ax, mode, metric, which=which
    )
    if reference_path is not None:
        fig, ax = _plot_optimizer_performance(
            reference_path, fig, ax, mode, metric, which=which
        )

    metrices = ["Test Loss", "Train Loss", "Test Accuracy", "Train Accuracy"]
    for idx, _metric in enumerate(metrices):
        # set y labels

        ax[idx].set_ylabel(_metric, fontsize=14)
        # rescale plots
        # ax[idx] = _rescale_ax(ax[idx])
        ax[idx].tick_params(labelsize=12)

    # show optimizer legends
    ax[0].legend(fontsize=12)

    ax[3].set_xlabel("Epochs", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.15)

    if show:
        plt.show()
    return fig, ax
