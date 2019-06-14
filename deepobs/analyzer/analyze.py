#!/usr/bin/env python

from __future__ import print_function
import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .analyze_utils import rescale_ax
from .. import config
from .analyze_utils import make_legend_and_colors_consistent

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
                testproblems[tp] = TestProblemAnalyzer(path, metric, self.reference_path)
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

# TODO implement lr sensitivity with default dict.get()
#    def plot_lr_sensitivity(self, baseline_pars=None, mode='final'):
#        print("Plot learning rate sensitivity plot")
#        # TODO make the plotting abstract for every testproblem and not only the fixed ones.
#        fig, axis = plt.subplots(2, 4, figsize=(35, 4))
#
#        # small testproblem set
#        ax_col = 0
#        for testprob in [
#                "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
#        ]:
#            if testprob in self.testproblems:
#                for _, opt in self.testproblems[testprob].optimizers.items(
#                ):
#                    opt.plot_lr_sensitivity(axis[0][ax_col], mode=mode)
#                ax_col += 1
#
#        # TODO wrap the baseline plotting somewhere else
#        if baseline_pars is not None:
#            ax_col = 0
#            for testprob in [
#                    "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
#            ]:
#                if testprob in baseline_pars.testproblems:
#                    for _, opt in baseline_pars.testproblems[
#                            testprob].optimizers.items():
#                        opt.plot_lr_sensitivity(axis[0][ax_col], mode=mode)
#                    ax_col += 1
#
#        # large testproblem set
#        ax_col = 0
#        for testprob in [
#                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"
#        ]:
#            if testprob in self.testproblems:
#                for _, opt in self.testproblems[testprob].optimizers.items(
#                ):
#                    opt.plot_lr_sensitivity(axis[1][ax_col], mode=mode)
#                ax_col += 1
#
#        # TODO same as above for baseline parser
#        if baseline_pars is not None:
#            ax_col = 0
#            for testprob in [
#                    "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164",
#                    "tolstoi_char_rnn"
#            ]:
#                if testprob in baseline_pars.testproblems:
#                    for _, opt in baseline_pars.testproblems[
#                            testprob].optimizers.items():
#                        opt.plot_lr_sensitivity(axis[1][ax_col], mode=mode)
#                    ax_col += 1
#
#        fig, axis = beautify_lr_sensitivity(
#            fig, axis)
#
#        # TODO implement textification
##        texify_lr_sensitivity(fig, axis)
#        plt.show()
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
    def __init__(self, path, metric, reference_path):
        """Initializes a new TestProblemAnalyzer instance.

        Args:
            path (str): Path to the parent folder of the test problem (i.e. the
                results folder).
            tp (str): Name of the test problem (same as the folder name).
        """
        self._path = path
        self.__name = path.split('/')[-1]
        self.metric = metric
        # TODO make the metrices attributes of the testproblem class?
        # TODO generalize this: if test accuracies not available, use test losses

        self.optimizers = self._read_optimizer(reference_path)

    def _read_optimizer(self, reference_path):
        """Read all optimizer (folders) in a test problem (folder).

        Returns:
            dict: Dictionary of optimizers, where the key is the optimizer's name
                and the value is an instance of the OptimizerAnalyzer class.

        """
        optimizers = dict()
        for opt in os.listdir(self._path):
            path = os.path.join(self._path, opt)
            optimizers[opt] = OptimizerAnalyzer(path, self.metric)

        # add the reference optimizers
        if reference_path is not None:
            reference_testproblem = os.path.join(reference_path, self.__name)
            for opt in os.listdir(reference_testproblem):
                            path = os.path.join(reference_testproblem, opt)
                            optimizers[opt] = OptimizerAnalyzer(path, self.metric)
        return optimizers

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
    def __init__(self, path, metric, conv_perf = 0.0):
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
        self._path = path
        self.__name = path.split('/')[-1]
        self.metric = metric
        # TODO if I call optimizer analyzer directly, the conv performance will always be 0?
        self.conv_perf = conv_perf
        self.__setting_analyzers = self.__read_setting_analyzers()
        self.num_settings = len(self.__setting_analyzers)
        self.best_SettingAnalyzer_final = self.__get_best_SettingAnalyzer_final()
        self.best_SettingAnalyzer_best = self.__get_best_SettingAnalyzer_best()
        self.most_run_SettingAnalyzer = self.__get_SettingAnalyzer_most_runs()

    def __read_setting_analyzers(self):
        """Read all settings (folders) in a optimizer (folder).

        Returns:
            dict: Dictionary of settings, where the key is the setting's name
                and the value is an instance of the SettingAnalyzer class.

        """
        settings = dict()
        for sett in os.listdir(self._path):
            path = os.path.join(self._path, sett)
            settings[sett] = SettingAnalyzer(path, self.metric, self.conv_perf)
        return settings

    def __get_best_SettingAnalyzer_final(self):
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

        for _, sett in self.__setting_analyzers.items():
            val = sett.final_value
            if better(val, current_best):
                current_best = val
                best_ind = sett
        return best_ind

    def __get_best_SettingAnalyzer_best(self):
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
        for _, sett in self.__setting_analyzers.items():
            val = sett.best_value
            if better(val, current_best):
                current_best = val
                best_ind = sett
        return best_ind

    def __get_SettingAnalyzer_most_runs(self):
        """Returns the setting with the most repeated runs (with the same
        setting, but probably different seeds).

        Returns:
            SettingAnalyzer: Instance of the SettingAnalyzer class with the most
            repeated runs.

        """
        most_runs = 0
        for _, sett in self.__setting_analyzers.items():
            if sett.num_runs > most_runs:
                most_runs = sett.num_runs
                most_run_setting = sett
        return most_run_setting

    def plot_optimizer_performance(self, axes=None, mode='most'):
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
        metrices = ['test_losses',
                    'train_losses',
                    'test_accuracies',
                    'train_accuracies']

        if axes is None:
            # TODO make the default axes prettier
            fig, axes = plt.subplots(4, 1, figsize=(25, 8))
            for idx, metric in enumerate(metrices):
                axes[idx].set_ylabel(metric)
            axes[3].set_xlabel("Epochs")

        if mode == 'final':
            sett = self.best_SettingAnalyzer_final
        elif mode == 'best':
            sett = self.best_SettingAnalyzer_best
        elif mode == 'most':
            sett = self.most_run_SettingAnalyzer
        else:
            raise RuntimeError("Mode unknown")

        for idx, metric in enumerate(metrices):
            if metric not in sett.aggregate:
                pass
            else:
                axes[idx].plot(
                    sett.aggregate[metric]['mean'],
                    label=self.__name)
                axes[idx].fill_between(
                    range(sett.aggregate[metric]['mean'].size),
                    sett.aggregate[metric]['mean'] -
                    sett.aggregate[metric]['std'],
                    sett.aggregate[metric]['mean'] +
                    sett.aggregate[metric]['std'],
                    color=axes[idx].get_lines()[-1].get_color(),
                    alpha=0.2)

# TODO how to plot the lr sens in the general case? pt vs tf? maybe preprocess the json settings?
#    def plot_lr_sensitivity(self, ax, mode='final'):
#        """Generates the ``learning rate`` sensitivity plot for this optimizer.
#        This plots the relative performance (relative to the best setting for
#        this optimizer) against the ``learning rate`` used in this setting.
#
#        This assumes that all settings or otherwise equal and only different in
#        the ``learning rate``.
#
#        Args:
#            ax (matplotlib.axes): Handle to a matplotlib axis to plot the
#                ``learning rate`` sensitivity onto.
#            mode (str): Whether to use the final (``final``) performance or the
#                best (``best``) when evaluating each setting.
#                Defaults to ``final``.
#        """
#        rel_perf = []
#        lr = []
#        for _, sett in self.setting_analyzers.items():
#
#            if mode == 'final':
#                val = sett.final_value
#                best = self.best_SettingAnalyzer_final.final_value
#            elif mode == 'best':
#                val = sett.best_value
#                best = self.best_SettingAnalyzer_best.best_value
#            else:
#                raise RuntimeError("Mode unknown")
#
#            if self.metric == 'test_losses' or self.metric == 'train_losses':
#                rel_perf.append(best / val)
#            elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
#                rel_perf.append(val / best)
#            else:
#                raise RuntimeError("Metric unknown")
#
#            lr.append(sett.settings['learning_rate'])
#        # TODO understand this piece
#        rel_perf = np.nan_to_num(rel_perf)  # replace NaN with zero
#        rel_perf = np.array(np.vstack((rel_perf, lr))).transpose()
#        rel_perf = rel_perf[rel_perf[:, 1].argsort()]
#        ax.plot(rel_perf[:, 1], rel_perf[:, 0], label=self.name)
#        ax.set_xscale('log')
#        ax.set_ylim([0.0, 1.0])
#
    # TODO bring performance dict to analyzer level
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
            sett = self.most_run_SettingAnalyzer
        else:
            raise RuntimeError("Mode unknown")

        perf_dict = dict()
        perf_dict['Performance'] = sett.aggregate[self.metric]['mean'][-1]
        perf_dict['Speed'] = sett.aggregate['speed']
        perf_dict['Tuneability'] = sett.settings
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
    def __init__(self, path, metric, conv_perf = 0.0):
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
        self._path = path
        self.metric = metric
        self.__runs = self.__read_all_runs()
        self.conv_perf = conv_perf
        # num_runs needs to be accessed by most_run analyzer
        self.num_runs = len(self.__runs)

        self.settings = self.__get_settings()
        # does aggregate need to be public?
        self.aggregate = self.__determine_aggregate_from_runs()
        self.final_value = self.__get_final_value()
        self.best_value = self.__get_best_value()

    def __get_final_value(self):
        """Get final (mean) value of the metric used in this test problem.
        Returns:
            float: Final (mean) value of the test problem's metric.
        """
        return self.aggregate[self.metric]['mean'][-1]

    def __get_best_value(self):
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

    def __get_settings(self):
        # all runs have the same setting, so just take the first run.
        # TODO should not raise an error if no run is available for this setting (i.e. folder is empty).
        # TODO write try catch to solve 0 run problem?
        json_data = self.__load_json(self._path, self.__runs[0])

        # captures all possible output setups at the moment
        # TODO once the final output design is fixed for both frameworks, simplify this
        settings = {**json_data.get('optimizer_hyperparams', {}),
                    **json_data.get('analyzable_training_params', {})}
        # TODO remove this workaround after final output design
        # workaround for old tensorflow outputs
        if 'learning_rate' in json_data:
            settings = {**settings, 'learning_rate': json_data['learning_rate']}
        return settings

    def __determine_aggregate_from_runs(self):
        # metrices
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for run in self.__runs:
            json_data = self.__load_json(self._path, run)
            train_losses.append(json_data['train_losses'])
            test_losses.append(json_data['test_losses'])
            # just add accuracies to the aggregate if they are available
            # in tensorflow they must exist, in pytorch they must be different from 0 (the default)
            if 'train_accuracies' in json_data and np.mean(json_data['train_accuracies'])!=0:
                train_accuracies.append(json_data['train_accuracies'])
                test_accuracies.append(json_data['test_accuracies'])

        aggregate = dict()

#       compute speed
        perf = np.array(eval(self.metric))
        if self.metric == "test_losses" or self.metric == "train_losses":
            # average over first time they reach conv perf (use num_epochs if conv perf is not reached)
            aggregate['speed'] = np.mean(
                np.argmax(perf <= self.conv_perf, axis=1) +
                np.invert(np.max(perf <= self.conv_perf, axis=1)) *
                perf.shape[1])
        elif self.metric == "test_accuracies" or self.metric == "train_accuracies":
            aggregate['speed'] = np.mean(
                np.argmax(perf >= self.conv_perf, axis=1) +
                np.invert(np.max(perf >= self.conv_perf, axis=1)) *
                perf.shape[1])

        for metrics in ['train_losses', 'test_losses', 'train_accuracies', 'test_accuracies']:
            # only add the metric if available
            if len(eval(metrics)) != 0:
            # exclude the parts which are NaN
                aggregate[metrics] = {
                    'mean': np.mean(eval(metrics), axis=0),
                    'std': np.std(eval(metrics), axis=0)
                }

        return aggregate

    def __read_all_runs(self):
        runs = [run for run in os.listdir(self._path) if run.endswith(".json")]
        return runs

    def __load_json(self, path, file_name):
        with open(os.path.join(path, file_name), "r") as f:
             json_data = json.load(f)
        return json_data