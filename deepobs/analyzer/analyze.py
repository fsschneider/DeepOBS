#!/usr/bin/env python

from __future__ import print_function
import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .. import config
from .shared_utils import _check_if_metric_is_available, create_setting_analyzer_ranking
from ..tuner.tuner_utils import generate_tuning_summary
from .analyze_utils import rescale_ax


class Analyzer:
    """DeepOBS analyzer class to generate result plots or get other summaries.

    Args:
        path (str): Path to the results folder. This folder should contain one
            or multiple testproblem folders.

    Attributes:
        testproblems: Dictionary of test problems where the key is the
            name of a test problem (e.g. ``cifar10_3c3d``) and the value is an
            instance of the TestProblemAnalyzer class (see below).
    """
    def __init__(self, results_path, metric = 'test_accuracies', reference_path = None):
        """Initializes a new Analyzer instance.

        Args:
            path (str): Path to the results folder. This folder should contain one
                or multiple testproblem folders.
        """
        self.metric = metric
        self.path = results_path
        self.reference_path = reference_path
        self.testproblems = self._read_testproblems()

    def _read_testproblems(self):
        """Read all test problems (folders) in this results folder.

        Returns:
            dict: Dictionary of test problems, where the key is the test
            problem's name and the value is an instance of the
            TestProblemAnalyzer class.

        """
        testproblems = dict()
        for tp in os.listdir(self.path):
            if tp == 'quadratic_deep' or tp == 'mnist_vae' or tp == 'fmnist_vae':
                metric = "test_losses"
            else:
                metric = self.metric

            path = os.path.join(self.path, tp)
            if os.path.isdir(path):
                print('Analyzing', tp)
        return testproblems

    def print_best_runs(self):
        runs_dict = self.get_best_runs()
        for tp_name, testproblem in self.testproblems.items():
            print('********************')
            print('Settings for ' + tp_name + ':')
            print('------------------')
            for opt_name , opt in testproblem.optimizers.items():
                print('\n' + opt_name + ':')
                # TODO flatten the dict of hyperparams for nice displayment
                print(pd.DataFrame(runs_dict[tp_name][opt_name]))
            print('********************')

#
#    def plot_table(self, baseline_pars=None):
#        print("Plot overall performance table")
#
#        bm_table_small = dict()
#        for testprob in [
#                "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
#        ]:
#            bm_table_small[testprob] = dict()
#            bm_table_small[testprob]['Performance'] = dict()
#            bm_table_small[testprob]['Speed'] = dict()
#            bm_table_small[testprob]['Tuneability'] = dict()
#            if testprob in self.testproblems:
#                for _, opt in self.testproblems[testprob].optimizers.items():
#                    bm_table_small[testprob] = opt.get_bm_table(
#                        bm_table_small[testprob])
#
#            if baseline_pars is not None:
#                if testprob in baseline_pars.testproblems:
#                    for _, opt in baseline_pars.testproblems[
#                            testprob].optimizers.items():
#                        bm_table_small[testprob] = opt.get_bm_table(
#                            bm_table_small[testprob])
#        bm_table_small_pd = beautify_plot_table(
#            bm_table_small)
#        texify_plot_table(bm_table_small_pd, "small")
#
#        bm_table_large = dict()
#        for testprob in [
#                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"
#        ]:
#            bm_table_large[testprob] = dict()
#            bm_table_large[testprob]['Performance'] = dict()
#            bm_table_large[testprob]['Speed'] = dict()
#            bm_table_large[testprob]['Tuneability'] = dict()
#            if testprob in self.testproblems:
#                for _, opt in self.testproblems[testprob].optimizers.items():
#                    bm_table_large[testprob] = opt.get_bm_table(
#                        bm_table_large[testprob])
#            if baseline_pars is not None:
#                if testprob in baseline_pars.testproblems:
#                    for _, opt in baseline_pars.testproblems[
#                            testprob].optimizers.items():
#                        bm_table_large[testprob] = opt.get_bm_table(
#                            bm_table_large[testprob])
#        bm_table_large_pd = beautify_plot_table(bm_table_large)
#        texify_plot_table(bm_table_large_pd,"large")

    def get_best_runs(self):
        """Iterates through all testproblems and optimizers to get the setting for each mode (best, final, most).
        Returns a nested dict structured as follows:
             {
             testproblem:{
                 optimizer: {
                     final: <a dict that contains the settings that led to the best final performance>,
                     best: <a dict that contains the settings that led to the best performance>
                     most: <a dict that contains the settings that hold the most runs>
                     }
                 ... <more optimizer>
                 }
            ...<more testproblems>
            }
        """
        best_runs = dict()
        for tp_name, testproblem in self.testproblems.items():
            opt_dic = dict()
            for opt_name, opt in testproblem.optimizers.items():
                best_setting_final = opt.best_SettingAnalyzer_final.settings
                best_setting_best = opt.best_SettingAnalyzer_best.settings
                setting_most = opt.most_run_SettingAnalyzer.settings
                opt_dic[opt_name] = {
                                'best' :  best_setting_best,
                                'final':  best_setting_final,
                                'most': setting_most
                                }
            best_runs[tp_name] = opt_dic
        return best_runs

    def plot_performance(self, mode='final'):

        # TODO if there are too many testproblems, split in two figures
        num_testproblems = len(self.testproblems)
        fig, axes = plt.subplots(4, num_testproblems, sharex='col', figsize=(25, 8))
        ax_col = 0
        for tp_name, tp in self.testproblems.items():
            for opt_name, opt in tp.optimizers.items():
                # workaround if there is only one testproblem
                if num_testproblems == 1:
                    opt.plot_optimizer_performance(axes, mode = mode)
                    axes[0].legend()
                else:
                    opt.plot_optimizer_performance(axes[:, ax_col], mode = mode)
                    axes[0,ax_col].legend()

            # rescaling
            if num_testproblems == 1:
                for idx, ax in enumerate(axes):
                    axes[idx] = rescale_ax(ax)
            else:
                for idx, ax in enumerate(axes[:, ax_col]):
                    axes[idx, ax_col] = rescale_ax(ax)

            ax_col += 1

        # all lines with the same label should have the same color such
        # that the color for the optimizer is the same for all testproblems
        # TODO make colors of legend consistent for all testproblems
#        make_legend_and_colors_consistent(axes)

        # label the plot
        rows = ['Test Loss', 'Train Loss', 'Test Accuracy', 'Train Accuracy']
        cols = [tp_name for tp_name, _ in self.testproblems.items()]

        # workaround if there is only one testproblem
        if num_testproblems == 1:
            axes[0].set_title(cols[0])
            for ax, row in zip(axes, rows):
                ax.set_ylabel(row)
        else:
            for ax, col in zip(axes[0], cols):
                ax.set_title(col)
            for ax, row in zip(axes[:,0], rows):
                ax.set_ylabel(row)

        plt.tight_layout()

        return fig, axes

    def _get_conv_perf(self):
        """Read the convergence performance for this test problem from a
        dictionary in the baseline folder.

        Returns:
            float: Convergence performance for this test problem

        """
        try:
            with open(os.path.join(config.get_baseline_dir(),
                         "convergence_performance.json"), "r") as f:
                return json.load(f)[self.__name]
        except IOError:
            print("Warning: Could not find a convergence performance for this testproblem. Either the file does not exist or there are no convergence results for this testproblem.")
            return 0.0


def plot_hyperparameter_sensitivity(optimizer_path, hyperparam, mode='final', metric = 'test_accuracies', xscale='linear'):
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


def get_performance_dictionary(optimizer_path, mode = 'final', metric = 'test_accuracies'):
    metric = _check_if_metric_is_available(optimizer_path, metric)
    setting_analyzers_ranking = create_setting_analyzer_ranking(optimizer_path, mode, metric)
    sett = setting_analyzers_ranking[0]

    perf_dict = dict()
    if mode == 'final':
        perf_dict['Performance'] = sett.final_value
    elif mode == 'best':
        perf_dict['Performance'] = sett.best_value
    else:
        raise RuntimeError('Mode not implemented for the performance dictionary')

    # TODO how and where compute speed?
    # perf_dict['Speed'] = sett.aggregate['speed']
    perf_dict['Tuneability'] = sett.aggregate['optimizer_hyperparams']
    return perf_dict


# TODO reference pathes
def plot_optimizer_performance(optimizer_path, mode = 'final', metric = 'test_accuracies'):
    metric = _check_if_metric_is_available(optimizer_path, metric)
    setting_analyzer_ranking = create_setting_analyzer_ranking(optimizer_path, mode, metric)
    setting = setting_analyzer_ranking[0]
    mean = setting.aggregate[setting.metric]['mean']
    std = setting.aggregate[setting.metric]['std']
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean-std,  mean+std, alpha=0.3)
    plt.show()


