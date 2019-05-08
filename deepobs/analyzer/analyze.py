#!/usr/bin/env python

from __future__ import print_function
import seaborn as sns
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib2tikz import save as tikz_save
from matplotlib2tikz import get_tikz_code

from .analyze_utils import beautify_lr_sensitivity
from .analyze_utils import beautify_plot_performance
from .analyze_utils import beautify_plot_table

# TODO do I really need seaborn?
sns.set()
sns.set_style("whitegrid", {
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False
})


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
    def __init__(self, results_path):
        """Initializes a new Analyzer instance.

        Args:
            path (str): Path to the results folder. This folder should contain one
                or multiple testproblem folders.
        """
        self.path = results_path
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
            if os.path.isdir(os.path.join(self.path, tp)):
                testproblems[tp] = TestProblemAnalyzer(self.path, tp)
        return testproblems

    def print_best_runs(self):
        runs_dict = self.get_best_runs()
        for _, testproblem in self.testproblems.items():
            print('********************')
            print('Settings for ' + testproblem.name + ':')
            print('------------------')
            for _ , opt in testproblem.optimizers.items():
                print('\n' + opt.name + ':')
                # TODO flatten the dict of hyperparams for nice displayment
                print(pd.DataFrame(runs_dict[testproblem.name][opt.name]))
            print('********************')

    def get_best_runs(self):
        # TODO rename since most does not neccessarily mean best run
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
        for _, testproblem in self.testproblems.items():
            opt_dic = dict()
            for _, opt in testproblem.optimizers.items():
                best_setting_final = opt.best_SettingAnalyzer_final.settings
                best_setting_best = opt.best_SettingAnalyzer_best.settings
                setting_most = opt.most_run_SettingAnalyzer.settings
                opt_dic[opt.name] = {
                                'best' :  best_setting_best,
                                'final':  best_setting_final,
                                'most': setting_most
                                }
            best_runs[testproblem.name] = opt_dic
        return best_runs
    # TODO print the best runs
    def plot_lr_sensitivity(self, baseline_pars=None, mode='final'):
        print("Plot learning rate sensitivity plot")
        # TODO make the plotting abstract for every testproblem and not only the fixed ones.
        fig, axis = plt.subplots(2, 4, figsize=(35, 4))

        # small testproblem set
        ax_col = 0
        for testprob in [
                "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
        ]:
            if testprob in self.testproblems:
                for _, opt in self.testproblems[testprob].optimizers.items(
                ):
                    opt.plot_lr_sensitivity(axis[0][ax_col], mode=mode)
                ax_col += 1

        # TODO wrap the baseline plotting somewhere else
        if baseline_pars is not None:
            ax_col = 0
            for testprob in [
                    "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
            ]:
                if testprob in baseline_pars.testproblems:
                    for _, opt in baseline_pars.testproblems[
                            testprob].optimizers.items():
                        opt.plot_lr_sensitivity(axis[0][ax_col], mode=mode)
                    ax_col += 1

        # large testproblem set
        ax_col = 0
        for testprob in [
                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"
        ]:
            if testprob in self.testproblems:
                for _, opt in self.testproblems[testprob].optimizers.items(
                ):
                    opt.plot_lr_sensitivity(axis[1][ax_col], mode=mode)
                ax_col += 1

        # TODO same as above for baseline parser
        if baseline_pars is not None:
            ax_col = 0
            for testprob in [
                    "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164",
                    "tolstoi_char_rnn"
            ]:
                if testprob in baseline_pars.testproblems:
                    for _, opt in baseline_pars.testproblems[
                            testprob].optimizers.items():
                        opt.plot_lr_sensitivity(axis[1][ax_col], mode=mode)
                    ax_col += 1

        fig, axis = beautify_lr_sensitivity(
            fig, axis)

        # TODO implement textification
#        texify_lr_sensitivity(fig, axis)
        plt.show()

    def plot_performance(self, baseline_pars=None, mode="most"):
        # Small Benchmark
        fig, axis = plt.subplots(4, 4, sharex='col', figsize=(25, 8))

        ax_col = 0
        for testprob in [
                "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
        ]:
            if testprob in self.testproblems:
                for _, opt in self.testproblems[testprob].optimizers.items(
                ):
                    opt.plot_performance(axis[:, ax_col], mode=mode)
                ax_col += 1

        if baseline_pars is not None:
            ax_col = 0
            for testprob in [
                    "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
            ]:
                if testprob in baseline_pars.testproblems:
                    for _, opt in baseline_pars.testproblems[
                            testprob].optimizers.items():
                        opt.plot_performance(axis[:, ax_col], mode='most')
                    ax_col += 1
        fig, axis = beautify_plot_performance(
            fig, axis, self, "small")
#        texify_plot_performance(fig, axis, "small")
        plt.show()

        # Large Benchmark
        fig, axis = plt.subplots(4, 4, sharex='col', figsize=(25, 8))

        ax_col = 0
        for testprob in [
                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"
        ]:
            if testprob in self.testproblems:
                for _, opt in self.testproblems[testprob].optimizers.items():
                    opt.plot_performance(axis[:, ax_col], mode=mode)
                ax_col += 1
        if baseline_pars is not None:
            ax_col = 0
            for testprob in [
                    "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164",
                    "tolstoi_char_rnn"
            ]:
                if testprob in baseline_pars.testproblems:
                    for _, opt in baseline_pars.testproblems[
                            testprob].optimizers.items():
                        opt.plot_performance(axis[:, ax_col], mode='most')
                    ax_col += 1
        fig, axis = beautify_plot_performance(
            fig, axis, self, "large")
        texify_plot_performance(fig, axis, "large")
        plt.show()

    def plot_table(self, baseline_pars=None):
        print("Plot overall performance table")

        bm_table_small = dict()
        for testprob in [
                "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
        ]:
            bm_table_small[testprob] = dict()
            bm_table_small[testprob]['Performance'] = dict()
            bm_table_small[testprob]['Speed'] = dict()
            bm_table_small[testprob]['Tuneability'] = dict()
            if testprob in self.testproblems:
                for _, opt in self.testproblems[testprob].optimizers.items():
                    bm_table_small[testprob] = opt.get_bm_table(
                        bm_table_small[testprob])

            if baseline_pars is not None:
                if testprob in baseline_pars.testproblems:
                    for _, opt in baseline_pars.testproblems[
                            testprob].optimizers.items():
                        bm_table_small[testprob] = opt.get_bm_table(
                            bm_table_small[testprob])
        bm_table_small_pd = beautify_plot_table(
            bm_table_small)
        texify_plot_table(bm_table_small_pd, "small")

        bm_table_large = dict()
        for testprob in [
                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"
        ]:
            bm_table_large[testprob] = dict()
            bm_table_large[testprob]['Performance'] = dict()
            bm_table_large[testprob]['Speed'] = dict()
            bm_table_large[testprob]['Tuneability'] = dict()
            if testprob in self.testproblems:
                for _, opt in self.testproblems[testprob].optimizers.items():
                    bm_table_large[testprob] = opt.get_bm_table(
                        bm_table_large[testprob])
            if baseline_pars is not None:
                if testprob in baseline_pars.testproblems:
                    for _, opt in baseline_pars.testproblems[
                            testprob].optimizers.items():
                        bm_table_large[testprob] = opt.get_bm_table(
                            bm_table_large[testprob])
        bm_table_large_pd = beautify_plot_table(bm_table_large)
        texify_plot_table(bm_table_large_pd,"large")

class TestProblemAnalyzer:
    """DeepOBS analyzer class for a specific test problem.

    This class will store all relevant information regarding a test problem,
    such as the convergence performance of this problem.

    Args:
        path (str): Path to the parent folder of the test problem (i.e. the
            results folder).
        tp (str): Name of the test problem (same as the folder name).

    Attributes:
        name: Name of the test problem in DeepOBS format
            (e.g. ``cifar10_3c3d``).
        conv_perf: Convergence performance for this test problem.
        metric: Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.
        optimizer: Dictionary of optimizers for this test problem where
            the key is the name of the optimizer (e.g.
            ``GradientDescentOptimizer``) and the value is an instance of the
            OptimizerAnalyzer class (see below).

    """
    def __init__(self, path, tp):
        """Initializes a new TestProblemAnalyzer instance.

        Args:
            path (str): Path to the parent folder of the test problem (i.e. the
                results folder).
            tp (str): Name of the test problem (same as the folder name).
        """
        self._path = os.path.join(path, tp)
        self.name = tp
        print("Analyzing", self.name)
        # TODO make the metrices attributes of the testproblem class?
        # TODO generalize this: if test accuracies not available, use test losses
        if tp == 'quadratic_deep' or tp == 'mnist_vae' or tp == 'fmnist_vae':
            self.metric = "test_losses"
        else:
            self.metric = "test_accuracies"
        self.optimizers = self._read_optimizer()

    def _read_optimizer(self):
        """Read all optimizer (folders) in a test problem (folder).

        Returns:
            dict: Dictionary of optimizers, where the key is the optimizer's name
                and the value is an instance of the OptimizerAnalyzer class.

        """
        optimizers = dict()
        for opt in os.listdir(self._path):
            optimizers[opt] = OptimizerAnalyzer(self._path, opt, self.metric, self.name)
        return optimizers

#    def _get_conv_perf(self):
#        """Read the convergence performance for this test problem from a
#        dictionary in the baseline folder.
#
#        Returns:
#            float: Convergence performance for this test problem
#
#        """
#        # TODO here is tf used!! This will not work for the pytorch version. Make the baseline dir a general config !
#        try:
#            with open(os.path.join(tensorflow.config.get_baseline_dir(),
#                         "convergence_performance.json"), "r") as f:
#                return json.load(f)[self.name]
#        except IOError:
#            print("Warning: Could not find a convergence performance file.")
#            return 0.0


class OptimizerAnalyzer:
    """DeepOBS analyzer class for an optimizer (and a specific test problem).

    This class will give access to all relevant information regarding this
    optimizer such as the best performing hyperparameter setting or the number
    of settings.

    Args:
        path (str): Path to the parent folder of the optimizer folder (i.e. the
            test problem folder).
        opt (str): Name of the optimizer (folder).
        metric (str): Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.
        testproblem (str): Name of the test problem this optimizer (folder)
            belongs to.
        conv_perf (float): Convergence performance of the test problem this
            optimizer (folder) belongs to.

    Attributes:
        name: Name of the optimizer (folder).
        metric: Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.
        testproblem: Name of the test problem this optimizer (folder)
            belongs to.
        conv_perf: Convergence performance for this test problem.
        settings: Dictionary of hyperparameter settings for this
            optimizer (on this test problem) where the key is the name of the
            setting (folder) and the value is an instance of the
            SettingAnalyzer class (see below).
        num_settings: Total number of settings for this optimizer
            (and test problem)
    """
    def __init__(self, path, opt, metric, testproblem):
        """Initializes a new OptimizerAnalyzer instance.

        Args:
            path (str): Path to the parent folder of the optimizer folder (i.e.
                the test problem folder).
            opt (str): Name of the optimizer (folder).
            metric (str): Metric to use for this test problem. If available this
                will be ``test_accuracies``, otherwise ``test_losses``.
            testproblem (str): Name of the test problem this optimizer (folder)
                belongs to.
            conv_perf (float): Convergence performance of the test problem this
                optimizer (folder) belongs to.

        """
        self._path = os.path.join(path, opt)
        self.name = opt
        self.metric = metric
        self.testproblem = testproblem
#        self.conv_perf = conv_perf
        self.setting_analyzers = self._read_setting_analyzers()
        self.num_settings = len(self.setting_analyzers)
        self.best_SettingAnalyzer_final = self._get_best_SettingAnalyzer_final()
        self.best_SettingAnalyzer_best = self._get_best_SettingAnalyzer_best()
        self.most_run_SettingAnalyzer = self._get_SettingAnalyzer_most_runs()

    def _read_setting_analyzers(self):
        """Read all settings (folders) in a optimizer (folder).

        Returns:
            dict: Dictionary of settings, where the key is the setting's name
                and the value is an instance of the SettingAnalyzer class.

        """
        settings = dict()
        for sett in os.listdir(self._path):
            settings[sett] = SettingAnalyzer(self._path, sett, self.metric,
                                           self.testproblem, self.name)
        return settings

    def _get_best_SettingAnalyzer_final(self):
        """Returns the setting for this optimizer that has the best final
        performance using the metric (``test_losses`` or ``test_accuracies``)
        defined for this test problem.

        Returns:
            SettingAnalyzer: Instance of the SettingAnalyzer class with the best
            final performance

        """
        if self.metric == 'test_losses' or self.metric == 'train_losses':
            current_best = np.inf
            better = lambda x, y: x < y
        elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
            current_best = -np.inf
            better = lambda x, y: x > y
        else:
            raise RuntimeError("Metric unknown")

        for _, sett in self.setting_analyzers.items():
            val = sett.final_value
            if better(val, current_best):
                current_best = val
                best_ind = sett
        return best_ind

    def _get_best_SettingAnalyzer_best(self):
        """Returns the setting for this optimizer that has the best overall
        performance using the metric (``test_losses`` or ``test_accuracies``)
        defined for this test problem. In contrast to ``get_best_setting_final``
        in not only looks at the final performance per setting, but the best
        performance per setting.

        Returns:
            SettingAnalyzer: Instance of the SettingAnalyzer class with the best
            overall performance

        """
        if self.metric == 'test_losses' or self.metric == 'train_losses':
            current_best = np.inf
            better = lambda x, y: x < y
        elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
            current_best = -np.inf
            better = lambda x, y: x > y
        else:
            raise RuntimeError("Metric unknown")
        for _, sett in self.setting_analyzers.items():
            val = sett.best_value
            if better(val, current_best):
                current_best = val
                best_ind = sett
        return best_ind

    def _get_SettingAnalyzer_most_runs(self):
        """Returns the setting with the most repeated runs (with the same
        setting, but probably different seeds).

        Returns:
            SettingAnalyzer: Instance of the SettingAnalyzer class with the most
            repeated runs.

        """
        most_runs = 0
        for _, sett in self.setting_analyzers.items():
            if sett.num_runs > most_runs:
                most_runs = sett.num_runs
                most_run_setting = sett
        return most_run_setting

    def plot_lr_sensitivity(self, ax, mode='final'):
        """Generates the ``learning rate`` sensitivity plot for this optimizer.
        This plots the relative performance (relative to the best setting for
        this optimizer) against the ``learning rate`` used in this setting.

        This assumes that all settings or otherwise equal and only different in
        the ``learning rate``.

        Args:
            ax (matplotlib.axes): Handle to a matplotlib axis to plot the
                ``learning rate`` sensitivity onto.
            mode (str): Whether to use the final (``final``) performance or the
                best (``best``) when evaluating each setting.
                Defaults to ``final``.
        """
        rel_perf = []
        lr = []
        for _, sett in self.setting_analyzers.items():

            if mode == 'final':
                val = sett.final_value
                best = self.best_SettingAnalyzer_final.final_value
            elif mode == 'best':
                val = sett.best_value
                best = self.best_SettingAnalyzer_best.best_value
            else:
                raise RuntimeError("Mode unknown")

            if self.metric == 'test_losses' or self.metric == 'train_losses':
                rel_perf.append(best / val)
            elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
                rel_perf.append(val / best)
            else:
                raise RuntimeError("Metric unknown")

            lr.append(sett.settings['learning_rate'])
        # TODO understand this piece
        rel_perf = np.nan_to_num(rel_perf)  # replace NaN with zero
        rel_perf = np.array(np.vstack((rel_perf, lr))).transpose()
        rel_perf = rel_perf[rel_perf[:, 1].argsort()]
        ax.plot(rel_perf[:, 1], rel_perf[:, 0], label=self.name)
        ax.set_xscale('log')
        ax.set_ylim([0.0, 1.0])

    def plot_performance(self, ax, mode='most'):
        """Generates a performance plot for this optimzer using one
        hyperparameter setting.

        Can either use the setting with the best final performance, the best
        overall performance or the setting with the most runs.

        This function will plot all four possible performance metrics
        (``test_losses``, ``train_losses``, ``test_accuracies`` and
        ``train_accuracies``).


        Args:
            ax (list): List of four matplotlib axis to plot the performancs
                metrics onto.
            mode (str): Whether to use the setting with the best final
                (``final``) performance, the best overall (``best``) performance
                or the setting with the most runs (``most``) when plotting.
                Defaults to ``most``.

        """
        if mode == 'final':
            sett = self.best_SettingAnalyzer_final
        elif mode == 'best':
            sett = self.best_SettingAnalyzer_best
        elif mode == 'most':
            sett = self.most_run_SettingAnalyzer
            print("Plotting", sett.num_runs, "runs for ", self.name,
                  "on", sett.testproblem)
        else:
            raise RuntimeError("Mode unknown")

        for idx, metric in enumerate([
                'test_losses', 'train_losses', 'test_accuracies',
                'train_accuracies'
        ]):
            ax[idx].plot(
                sett.aggregate[metric]['mean'],
                label=sett.optimizer)
            ax[idx].fill_between(
                range(sett.aggregate[metric]['mean'].size),
                sett.aggregate[metric]['mean'] -
                sett.aggregate[metric]['std'],
                sett.aggregate[metric]['mean'] +
                sett.aggregate[metric]['std'],
                color=ax[idx].get_lines()[-1].get_color(),
                alpha=0.2)

    def get_bm_table(self, perf_table, mode='most'):
        """Generates the overall performance table for this optimizer.
        This includes metrics for the performance, speed and tuneability of this
        optimizer (on this test problem).
        Args:
            perf_table (dict): A dictionary with three keys: ``Performance``,
                ``Speed`` and ``Tuneability``.
            mode (str): Whether to use the setting with the best final
                (``final``) performance, the best overall (``best``) performance
                or the setting with the most runs (``most``).
                Defaults to ``most``.
        Returns:
            dict: Dictionary with holding the performance, speed and tuneability
            measure for this optimizer.
        """
        if mode == 'final':
            sett = self.best_SettingAnalyzer_final
        elif mode == 'best':
            sett = self.best_SettingAnalyzer_best
        elif mode == 'most':
            sett = self.most_run_SettingAnalyzer

        perf_table['Performance'][self.name] = sett.aggregate[
            self.metric]['mean'][-1]
        # TODO include speed
#        perf_table['Speed'][self.name] = sett.aggregate['speed']
        perf_table['Tuneability'][self.name] = {
            **{
                'lr': '{:0.2e}'.format(sett.settings['learning_rate'])
            },
            **sett.settings['hyperparams']
        }
        return perf_table

    def get_performance_dictionary(self, mode='most'):
        """Generates the overall performance overview for this optimizer.

        This includes metrics for the performance, speed and tuneability of this
        optimizer (on this test problem).

        Args:
            perf_table (dict): A dictionary with three keys: ``Performance``,
                ``Speed`` and ``Tuneability``.
            mode (str): Whether to use the setting with the best final
                (``final``) performance, the best overall (``best``) performance
                or the setting with the most runs (``most``).
                Defaults to ``most``.

        Returns:
            dict: Dictionary with holding the performance, speed and tuneability
            measure for this optimizer.

        """
        if mode == 'final':
            sett = self.best_SettingAnalyzer_final
        elif mode == 'best':
            sett = self.best_SettingAnalyzer_best
        elif mode == 'most':
            sett = self.SettingAnalyzer_most_runs
        else:
            raise RuntimeError("Mode unknown")

        perf_dict = dict()
        perf_dict['Performance'][self.name] = sett.aggregate[
            self.metric]['mean'][-1]
        perf_dict['Speed'][self.name] = sett.aggregate['speed']
        perf_dict['Tuneability'][self.name] = {
            **{
                'lr': '{:0.2e}'.format(sett.settings['learning_rate'])
            },
            **sett.settings['hyperparams']
        }
        return perf_dict


class SettingAnalyzer:
    """DeepOBS analyzer class for a setting (a hyperparameter setting).

    Args:
        path (str): Path to the parent folder of the setting folder (i.e. the
            optimizer folder).
        sett (str): Name of the setting (folder).
        metric (str): Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.
        testproblem (str): Name of the test problem this setting (folder)
            belongs to.
        conv_perf (float): Convergence performance of the test problem this
            setting (folder) belongs to.

    Attributes:
        name (str): Name of the setting (folder).
        metric (str): Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.
        testproblem (str): Name of the test problem this setting (folder)
            belongs to.
        conv_perf (float): Convergence performance for this test problem.
        aggregate (dictionary): Contains the mean and std of the runs for the given metric.
        runs (list): A list of all .json files for this setting, i.e. a list of all run results.
        settings (dictionary): Contains all settings that were relevant for the runs (batch size, learning rate, hyperparameters of the optimizer, etc). Random seed is not included.
        num_runs (int): The number of runs or this setting (most likely because of different random seeds)
    """
    def __init__(self, path, sett, metric, testproblem, optimizer):
        """Initializes a new SettingAnalyzer instance.

        Args:
            name (str): Path to the parent folder of the setting folder (i.e. the
                optimizer folder).
            sett (str): Name of the setting (folder).
            metric (str): Metric to use for this test problem. If available this
                will be ``test_accuracies``, otherwise ``test_losses``.
            testproblem (str): Name of the test problem this setting (folder)
                belongs to.
            conv_perf (float): Convergence performance of the test problem this
                setting (folder) belongs to.
        """
        self._path = os.path.join(path, sett)
        self.name = sett
        self.metric = metric
        self.testproblem = testproblem
        self.optimizer = optimizer
#        self.conv_perf = conv_perf
        self.runs = self._get_all_runs()
        self.num_runs = len(self.runs)
        self.settings = self._get_settings()
        self.aggregate = self._determine_aggregate_from_runs()
        self.final_value = self._get_final_value()
        self.best_value = self._get_best_value()

    def _get_final_value(self):
        """Get final (mean) value of the metric used in this test problem.
        Returns:
            float: Final (mean) value of the test problem's metric.
        """
        return self.aggregate[self.metric]['mean'][-1]

    def _get_best_value(self):
        """Get best (mean) value of the metric used in this test problem.
        Returns:
            float: Best (mean) value of the test problem's metric.
        """
        if self.metric == 'test_losses' or self.metric == 'train_losses':
            return min(self.aggregate[self.metric]['mean'])
        elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
            return max(self.aggregate[self.metric]['mean'])
        else:
            raise RuntimeError("Metric unknown")

    def _get_settings(self):
        # all runs have the same setting, so just take the first run.
        # TODO should not raise an error if no run is available for this setting (i.e. folder is empty).
        # make the interested settings a config global variable?
        json_data = self._load_json(os.path.join(self._path, self.runs[0]))
        settings = {k: json_data[k] for k in ['learning_rate', 'hyperparams']}
        return settings

    def _determine_aggregate_from_runs(self):
        # metrices
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for run in self.runs:
            json_data = self._load_json(os.path.join(self._path, run))
            train_losses.append(json_data['train_losses'])
            test_losses.append(json_data['test_losses'])
            if 'train_accuracies' in json_data:
                train_accuracies.append(json_data['train_accuracies'])
                test_accuracies.append(json_data['test_accuracies'])

        aggregate = dict()
        # compute speed
#        perf = np.array(eval(self.metric))
#        if self.metric == "test_losses" or self.metric == "train_losses":
#            # average over first time they reach conv perf (use num_epochs if conv perf is not reached)
#            aggregate['speed'] = np.mean(
#                np.argmax(perf <= self.conv_perf, axis=1) +
#                np.invert(np.max(perf <= self.conv_perf, axis=1)) *
#                perf.shape[1])
#        elif self.metric == "test_accuracies" or self.metric == "train_accuracies":
#            aggregate['speed'] = np.mean(
#                np.argmax(perf >= self.conv_perf, axis=1) +
#                np.invert(np.max(perf >= self.conv_perf, axis=1)) *
#                perf.shape[1])
        for metrics in ['train_losses', 'test_losses', 'train_accuracies', 'test_accuracies']:
            aggregate[metrics] = {
                'mean': np.mean(eval(metrics), axis=0),
                'std': np.std(eval(metrics), axis=0)
            }

        return aggregate

    def _get_all_runs(self):
        runs = [run for run in os.listdir(self._path) if run.endswith(".json")]
        return runs

    def _load_json(self, path):
        """Load the ``JSON`` file of the given path.

        Args:
            path (str): Path to a ``JSON`` file.

        Returns:
            dict: Dictionary from the ``JSON`` file.

        """
        with open(path, "r") as f:
            return json.load(f)