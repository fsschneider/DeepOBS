import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
from matplotlib2tikz import get_tikz_code
from .. import tensorflow


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
    def __init__(self, path):
        """Initializes a new Analyzer instance.

        Args:
            path (str): Path to the results folder. This folder should contain one
                or multiple testproblem folders.
        """
        self.path = path
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
        print("Setting up", self.name)
        self.conv_perf = self._get_conv_perf()
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
            optimizers[opt] = OptimizerAnalyzer(self._path, opt, self.metric,
                                              self.name, self.conv_perf)
        return optimizers

    def _get_conv_perf(self):
        """Read the convergence performance for this test problem from a
        dictionary in the baseline folder.

        Returns:
            float: Convergence performance for this test problem

        """
        try:
            with open(os.path.join(tensorflow.config.get_baseline_dir(),
                         "convergence_performance.json"), "r") as f:
                return json.load(f)[self.name]
        except IOError:
            print("Warning: Could not find a convergence performance file.")
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
    def __init__(self, path, opt, metric, testproblem, conv_perf):
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
        self.conv_perf = conv_perf
        self.settings = self._read_settings()
        self.num_settings = len(self.settings)
        self._best_setting_final = None
        self._best_setting_best = None

    def _read_settings(self):
        """Read all settings (folders) in a optimizer (folder).

        Returns:
            dict: Dictionary of settings, where the key is the setting's name
                and the value is an instance of the SettingAnalyzer class.

        """
        settings = dict()
        for sett in os.listdir(self._path):
            settings[sett] = SettingAnalyzer(self._path, sett, self.metric,
                                           self.testproblem, self.conv_perf)
        return settings

    def get_best_setting_final(self):
        """Returns the setting for this optimizer that has the best final
        performance using the metric (``test_losses`` or ``test_accuracies``)
        defined for this test problem.

        Returns:
            SettingAnalyzer: Instance of the SettingAnalyzer class with the best
            final performance

        """
        if self._best_setting_final is not None:
            return self._best_setting_final
        else:
            if self.metric == 'test_losses' or self.metric == 'train_losses':
                current_best = np.inf
                better = lambda x, y: x < y
            elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
                current_best = -np.inf
                better = lambda x, y: x > y
            else:
                raise RuntimeError("Metric unknown")
            best_sett = None
            for _, sett in self.settings.items():
                val = sett.aggregate.final_value
                if better(val, current_best):
                    current_best = val
                    best_ind = sett
            self._best_setting_final = best_ind
            return best_ind

    def get_best_setting_best(self):
        """Returns the setting for this optimizer that has the best overall
        performance using the metric (``test_losses`` or ``test_accuracies``)
        defined for this test problem. In contrast to ``get_best_setting_final``
        in not only looks at the final performance per setting, but the best
        performance per setting.

        Returns:
            SettingAnalyzer: Instance of the SettingAnalyzer class with the best
            overall performance

        """
        if self._best_setting_best is not None:
            return self._best_setting_best
        else:
            if self.metric == 'test_losses' or self.metric == 'train_losses':
                current_best = np.inf
                better = lambda x, y: x < y
            elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
                current_best = -np.inf
                better = lambda x, y: x > y
            else:
                raise RuntimeError("Metric unknown")
            best_sett = None
            for _, sett in self.settings.items():
                val = sett.aggregate.best_value
                if better(val, current_best):
                    current_best = val
                    best_ind = sett
            self._best_setting_best = best_ind
            return best_ind

    def get_setting_most_runs(self):
        """Returns the setting with the most repeated runs (with the same
        setting, but probably different seeds).

        Returns:
            SettingAnalyzer: Instance of the SettingAnalyzer class with the most
            repeated runs.

        """
        most_runs = 0
        for _, sett in self.settings.items():
            if sett.aggregate.num_runs > most_runs:
                most_runs = sett.aggregate.num_runs
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
        for _, sett in self.settings.items():
            if mode == 'final':
                val = sett.aggregate.final_value
                best = self.get_best_setting_final().aggregate.final_value
            elif mode == 'best':
                val = sett.aggregate.best_value
                best = self.get_best_setting_best().aggregate.best_value
            else:
                raise RuntimeError("Mode unknown")
            if self.metric == 'test_losses' or self.metric == 'train_losses':
                rel_perf.append(best / val)
            elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
                rel_perf.append(val / best)
            else:
                raise RuntimeError("Metric unknown")
            lr.append(sett.aggregate.output['learning_rate'])
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
            run = self.get_best_setting_final()
        elif mode == 'best':
            run = self.get_best_setting_best()
        elif mode == 'most':
            run = self.get_setting_most_runs()
            print("Plotting", run.aggregate.num_runs, "runs for ", self.name,
                  "on", run.aggregate.output['testproblem'])
        else:
            raise RuntimeError("Mode unknown")
        for idx, metric in enumerate([
                'test_losses', 'train_losses', 'test_accuracies',
                'train_accuracies'
        ]):
            ax[idx].plot(
                run.aggregate.output[metric]['mean'],
                label=run.aggregate.output['optimizer'])
            ax[idx].fill_between(
                range(run.aggregate.output[metric]['mean'].size),
                run.aggregate.output[metric]['mean'] -
                run.aggregate.output[metric]['std'],
                run.aggregate.output[metric]['mean'] +
                run.aggregate.output[metric]['std'],
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
            run = self.get_best_setting_final()
        elif mode == 'best':
            run = self.get_best_setting_best()
        elif mode == 'most':
            run = self.get_setting_most_runs()
        perf_table['Performance'][self.name] = run.aggregate.output[
            self.metric]['mean'][-1]
        perf_table['Speed'][self.name] = run.aggregate.output['speed']
        perf_table['Tuneability'][self.name] = {
            **{
                'lr': '{:0.2e}'.format(run.aggregate.output['learning_rate'])
            },
            **run.aggregate.output['hyperparams']
        }
        return perf_table


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
        aggregate (AggregateRun): Instance of the AggregateRun class for all
            runs with this setting.
    """
    def __init__(self, path, sett, metric, testproblem, conv_perf):
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
        self.conv_perf = conv_perf
        self.aggregate = self._get_aggregate()

    def _get_aggregate(self):
        """Create aggregate run for all runs in this setting folder.

        Returns:
            AggregateRun: Instance of the AggregateRun class holding the
            aggregate information of all runs with these settings.

        """
        runs = []
        for r in os.listdir(self._path):
            if r.endswith(".json"):
                runs.append(r)
        return AggregateRun(self._path, runs, self.name, self.metric,
                            self.testproblem, self.conv_perf)


class AggregateRun:
    """DeepOBS class for a group of runs witht the same settings (but possibly
    different seeds).

    Args:
        path (str): Path to the parent folder of the aggregate run folder (i.e.
            the settings folder).
        runs (list): List of run names all with the same setting.
        name (str): Name of the aggregate run (folder).
        metric (str): Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.
        testproblem (str): Name of the test problem this aggregate run (folder)
            belongs to.
        conv_perf (float): Convergence performance of the test problem this
            aggregate run (folder) belongs to.

    Attributes:
        name: Name of the aggregate run (folder).
        testproblem: Name of the test problem this aggregate run (folder)
            belongs to.
        conv_perf: Convergence performance for this test problem.
        runs: List of run names all with the same setting.
        num_runs: Number of runs (with the same setting).
        metric: Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.
        output: Dictionary including all aggregate information of the
            runs with this setting. All performance metrics have a mean and a
            standard deviation (can be zero if there is only one run with this
            setting).
        final_value: Final (mean) value of the test problem's metric
        best_value: Best (mean) value of the test problem's metric
    """
    def __init__(self, path, runs, name, metric, testproblem, conv_perf):
        """Initializes a new AggregateRun class.

        Args:
            path (str): Path to the parent folder of the aggregate run folder (i.e.
                the settings folder).
            runs (list): List of run names all with the same setting.
            name (str): Name of the aggregate run (folder).
            metric (str): Metric to use for this test problem. If available this
                will be ``test_accuracies``, otherwise ``test_losses``.
            testproblem (str): Name of the test problem this aggregate run (folder)
                belongs to.
            conv_perf (float): Convergence performance of the test problem this
                aggregate run (folder) belongs to.
        """
        self._path = path
        self.name = name
        self.testproblem = testproblem
        self.conv_perf = conv_perf
        self.runs = runs
        self.num_runs = len(runs)
        self.metric = metric
        self.output = self._aggregate()
        self.final_value = self._get_final_value()
        self.best_value = self._get_best_value()

    def _aggregate(self):
        """Aggregate performance data over all runs.

        Returns:
            dict: Dictionary including all aggregate information of the
            runs with this setting. All performance metrics have a mean and a
            standard deviation (can be zero if there is only one run with this
            setting).

        """
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        meta_loaded = False
        for run in self.runs:
            output = self._load_json(os.path.join(self._path, run))
            # Get meta data from first run
            if not meta_loaded:
                meta = output
                meta_loaded = True
            train_losses.append(output['train_losses'])
            test_losses.append(output['test_losses'])
            if 'train_accuracies' in output:
                train_accuracies.append(output['train_accuracies'])
                test_accuracies.append(output['test_accuracies'])
        aggregate = dict()
        # compute speed
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
        # build dict
        for m in [
                'train_losses', 'test_losses', 'train_accuracies',
                'test_accuracies'
        ]:
            aggregate[m] = {
                'mean': np.mean(eval(m), axis=0),
                'std': np.std(eval(m), axis=0)
            }
        # merge meta and aggregate (aggregate replaces)
        aggregate = {**meta, **aggregate}
        aggregate.pop('minibatch_train_losses', None)
        return aggregate

    def _load_json(self, path):
        """Load the ``JSON`` file of the given path.

        Args:
            path (str): Path to a ``JSON`` file.

        Returns:
            dict: Dictionary from the ``JSON`` file.

        """
        with open(path, "r") as f:
            return json.load(f)

    def _get_final_value(self):
        """Get final (mean) value of the metric used in this test problem.

        Returns:
            float: Final (mean) value of the test problem's metric.

        """
        return self.output[self.metric]['mean'][-1]

    def _get_best_value(self):
        """Get best (mean) value of the metric used in this test problem.

        Returns:
            float: Best (mean) value of the test problem's metric.

        """
        if self.metric == 'test_losses' or self.metric == 'train_losses':
            return min(self.output[self.metric]['mean'])
        elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
            return max(self.output[self.metric]['mean'])
        else:
            raise RuntimeError("Metric unknown")


def beautify_lr_sensitivity(fig, ax):
    """Beautify a learning rate sensitivity plot.

    This function adds axis labels and removes spines to create a nicer learning
    rate sensitivity plot.

    Args:
        fig (matplotlib.figure): Handle to the matplotlib figure of the learning
            rate sensitivity plot.
        ax (list): List of lists of matplotlib axis of the learning rate
            sensitivity plots.

    Returns:
        matplotlib.figure: Handle to the beautified matplotlib figure of the
        learning rate sensitivity plot.
        list: List of lists of the beautified matplotlib axis of the learning
        rate sensitivity plots.

    """
    fig.suptitle("Learning rate sensitivity", fontsize=20)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i][j].get_yaxis().set_visible(False)
            ax[i][j].spines['top'].set_visible(False)
            ax[i][j].spines['right'].set_visible(False)
            #     ax[i][j].spines['bottom'].set_visible(False)
            ax[i][j].spines['left'].set_visible(False)
            if i == 0:
                ax[i][j].get_xaxis().set_visible(False)
            if i == 1:
                ax[i][j].set_xlabel('Learning Rate')
    return fig, ax


def texify_lr_sensitivity(fig, ax):
    """Write a ``.tex`` file with the learning rate sensitivity plot.

    The function will create a file named `tuning_plot.tex` with the latex code
    for the learning rate sensitivity plot.

    Args:
        fig (matplotlib.figure): Handle to the matplotlib figure of the learning
            rate sensitivity plot.
        ax (list): List of lists of matplotlib axis of the learning rate
            sensitivity plots.

    Returns:
        str: String of the latex code for the learning rate sensitivity plot.

    """
    tikz_code = get_tikz_code(
        'tuning_plot_new.tex',
        figureheight='\\figureheight',
        figurewidth='0.33\\figurewidth')

    tikz_code = tikz_code.replace(
        '\\begin{groupplot}[group style={group size=4 by 2}]',
        '\\begin{groupplot}[group style={group size=4 by 2, horizontal sep=0.02\\figurewidth, vertical sep=0.15cm}]'
    )
    tikz_code = r"\pgfplotsset{every axis/.append style={label style={font=\tiny}, tick label style={font=\tiny}, legend style={font=\tiny, line width=1pt}}}" + tikz_code
    tikz_code = tikz_code.replace('minor', '%minor')  # comment minor tick
    tikz_code = tikz_code.replace('x grid',
                                  '%x grid')  # remove grid xmajorticks=false,
    tikz_code = tikz_code.replace('y grid', '%y grid')  # remove grid
    tikz_code = tikz_code.replace('tick align',
                                  '%tick align')  # ugly outside ticks
    tikz_code = tikz_code.replace(
        'nextgroupplot[', 'nextgroupplot[axis x line*=bottom,\nhide y axis,'
    )  # ugly outside ticks
    tikz_code = tikz_code.replace(
        '(current bounding box.south west)!0.98!(current bounding box.north west)',
        '(current bounding box.south west)!1.05!(current bounding box.north west)'
    )  # position title higher
    tikz_code = tikz_code.replace('title={',
                                  'title={\small ')  # shrink title size

    # Write the file out again
    with open('tuning_plot.tex', 'w') as file:
        file.write(tikz_code)

    return tikz_code


def rescale_ax(ax):
    """Rescale an axis to include the most important data.

    Args:
        ax (matplotlib.axis): Handle to a matplotlib axis.

    """
    lines = ax.lines
    y_data = []
    y_limits = []
    for line in lines:
        if line.get_label() != "convergence_performance":
            y_data.append(line.get_ydata())
        else:
            y_limits.append(line.get_ydata()[0])
    if y_data:
        y_limits.append(np.percentile(np.array(y_data), 20))
        y_limits.append(np.percentile(np.array(y_data), 80))
        y_limits = y_limits + (np.array(y_data)[:, -1].tolist())
        y_limits = [np.min(y_limits), np.max(y_limits)]
        y_limits = [y_limits[0] * 0.9, y_limits[1] * 1.1]
        if y_limits[0] != y_limits[1]:
            ax.set_ylim([max(1e-10, y_limits[0]), y_limits[1]])
        ax.margins(x=0)
    else:
        ax.set_ylim([1.0, 2.0])


def beautify_plot_performance(fig, ax, folder_parser, problem_set):
    """Beautify a performance plot.

    This function adds axis labels, sets titles and more to create a nicer
    performance plot.

    Args:
        fig (matplotlib.figure): Handle to the matplotlib figure of the
            performance plot.
        ax (list): List of lists of matplotlib axis of the performance plot.
        folder_parser (Analyzer): An instance of the DeepOBS Analyzer class
            to plot the performance from.
        problem_set (str): Can either be ``small`` or ``large`` to switch
            between which benchmark set is being plotted.

    Returns:
        matplotlib.figure: Handle to the beautified matplotlib figure of the
        performance plot.
        list: List of lists of the beautified matplotlib axis of the performance
        plots.

    """
    fig.subplots_adjust(hspace=0.4)
    if problem_set == "small":
        fig.suptitle("Benchmark Set Small", fontsize=20)
        titles = [
            "P1 Quadratic Deep", "P2 MNIST - VAE", "P3 F-MNIST - CNN",
            "P4 CIFAR-10 - CNN"
        ]
        # clear axis (needed for matplotlib2tikz)
        plt.sca(ax[2][0])
        plt.cla()
        plt.sca(ax[2][1])
        plt.cla()
        plt.sca(ax[3][1])
        plt.cla()
        ax[2][1].axis('off')
        ax[3][1].axis('off')
        ax[1][0].set_xlabel("Epochs")
        ax[1][1].set_xlabel("Epochs")
        ax[2][2].set_ylabel("Test Accuracy")
        ax[3][2].set_ylabel("Train Accuracy")
        ax[1][1].tick_params(
            axis='x', which='major', bottom=False,
            labelbottom=True)  # show x axis
        # Add convergence performance line
        for idx, tp in enumerate(
            ["quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"]):
            if tp in folder_parser.testproblems:
                metric = folder_parser.testproblems[tp].metric
                conv_perf = folder_parser.testproblems[tp].conv_perf
                if metric == "test_losses":
                    ax_row = 0
                elif metric == "test_accuracies":
                    ax_row = 2
                ax[ax_row][idx].axhline(
                    conv_perf, color='#AFB3B7', label="convergence_performance")
    elif problem_set == "large":
        fig.suptitle("Benchmark Set Large", fontsize=20)
        ax[1][0].set_xlabel("Epochs")
        ax[3][1].set_xlabel("Epochs")
        ax[2][1].set_ylabel("Test Accuracy")
        ax[3][1].set_ylabel("Train Accuracy")
        titles = [
            "P5 F-MNIST - VAE", "P6 CIFAR 100 - All CNN C",
            "P7 SVHN - Wide ResNet 16-4", "P8 Tolstoi - Char RNN"
        ]
        # Add convergence performance line
        for idx, tp in enumerate([
                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164",
                "tolstoi_char_rnn"
        ]):
            if tp in folder_parser.testproblems:
                metric = folder_parser.testproblems[tp].metric
                conv_perf = folder_parser.testproblems[tp].conv_perf
                if metric == "test_losses":
                    ax_row = 0
                elif metric == "test_accuracies":
                    ax_row = 2
                ax[ax_row][idx].axhline(
                    conv_perf, color='#AFB3B7', label="convergence_performance")
    # clear axis (needed for matplotlib2tikz)
    plt.sca(ax[2][0])
    plt.cla()
    plt.sca(ax[3][0])
    plt.cla()
    ax[2][0].axis('off')
    ax[3][0].axis('off')
    ax[3][2].set_xlabel("Epochs")
    ax[3][3].set_xlabel("Epochs")
    ax[0][0].set_ylabel("Test Loss")
    ax[1][0].set_ylabel("Train Loss")
    ax[1][0].tick_params(
        axis='x', which='major', bottom=False, labelbottom=True)  # show x axis
    # automatic rescaling
    for axlist in ax:
        for a in axlist:
            a = rescale_ax(a)
    # Legend
    handles, labels = ax[0][3].get_legend_handles_labels()
    #     labels_tex = [tfobs.plot_utils.texify(l) for l in labels]
    ax[3][0].legend(
        handles,
        labels,
        loc='upper right',
        bbox_to_anchor=(0.2, 1.1, 0.5, 0.5))
    for idx, title in enumerate(titles):
        ax[0, idx].set_title(title)
    return fig, ax


def texify_plot_performance(fig, ax, problem_set):
    """Write a ``.tex`` file with the performance plot.

    The function will create a file named `benchmark_small.tex` or
    `benchmark_large.tex` with the latex code for the performance plot.

    Args:
        fig (matplotlib.figure): Handle to the matplotlib figure of the
            performance plot.
        ax (list): List of lists of matplotlib axis of the performance plot.
        problem_set (str): Can either be ``small`` or ``large`` to switch
            between which benchmark set is being plotted.

    Returns:
        str: String of the latex code for the learning rate sensitivity plot.

    """
    file_name = 'benchmark_' + str(problem_set) + '.tex'
    tikz_code = get_tikz_code(
        file_name, figureheight='\\figureheight', figurewidth='\\figurewidth')

    tikz_code = r"\pgfplotsset{every axis/.append style={label style={font=\tiny}, tick label style={font=\tiny}, legend style={font=\tiny, line width=1pt}}}" + tikz_code
    tikz_code = tikz_code.replace('minor', '%minor')  # comment minor tick
    tikz_code = tikz_code.replace('x grid', '%x grid')  # remove grid
    tikz_code = tikz_code.replace('y grid', '%y grid')  # remove grid
    tikz_code = tikz_code.replace('tick align',
                                  '%tick align')  # ugly outside ticks
    tikz_code = tikz_code.replace(
        'nextgroupplot[',
        'nextgroupplot[axis x line*=bottom,\naxis y line*=left,'
    )  # ugly outside ticks
    tikz_code = tikz_code.replace('xlabel={Epochs},\nxmajorticks=false,',
                                  'xlabel={Epochs},\nxmajorticks=true,'
                                  )  # if x label is epoch, show ticks
    tikz_code = tikz_code.replace('ymajorticks=false,',
                                  'ymajorticks=true,')  # show y labels
    tikz_code = tikz_code.replace('\mathdefault',
                                  '')  # remove mathdefault in labels
    tikz_code = tikz_code.replace(
        '\path [draw=white!80.0!black, fill opacity=0]',
        '%\path [draw=white!80.0!black, fill opacity=0]'
    )  # remove lines that are created for some reason
    tikz_code = tikz_code.replace(
        '(current bounding box.south west)!0.98!(current bounding box.north west)',
        '(current bounding box.south west)!1.05!(current bounding box.north west)'
    )  # position title higher
    tikz_code = tikz_code.replace('title={',
                                  'title={\small ')  # shrink title size
    tikz_code = tikz_code.replace(
        'group style={group size=4 by 4',
        'group style={group size=4 by 4, horizontal sep=1cm, vertical sep=0.4cm '
    )  # reduce separation between plots
    tikz_code = tikz_code.replace(
        'ylabel={Test Loss}', r'ylabel style={align=left}, ylabel=Test\\Loss'
    )  # y label in two lines
    tikz_code = tikz_code.replace(
        'ylabel={Test Accuracy}',
        r'ylabel style={align=left}, ylabel=Test\\Accuracy'
    )  # y label in two lines
    tikz_code = tikz_code.replace(
        'ylabel={Train Loss}', r'ylabel style={align=left}, ylabel=Train\\Loss'
    )  # y label in two lines
    tikz_code = tikz_code.replace(
        'ylabel={Train Accuracy}',
        r'ylabel style={align=left}, ylabel=Train\\Accuracy'
    )  # y label in two lines

    # Write the file out again
    with open(file_name, 'w') as file:
        file.write(tikz_code)

    return tikz_code


def beautify_plot_table(bm_table):
    """Beautify a performance table.

    This function makes a few changes to the performance table to make it nicer.

    Args:
        bm_table (dict): Dictionary holding all the information for the
            performance table.

    Returns:
        pandas.dataframe: A pandas data frame for the performance table.
    """
    bm_table_pd = pd.DataFrame.from_dict({(i, j): bm_table[i][j]
                                          for i in bm_table.keys()
                                          for j in bm_table[i].keys()}).T
    cols = list(bm_table_pd.columns.values)
    if 'AdamOptimizer' in cols:
        cols.insert(0, cols.pop(cols.index('AdamOptimizer')))
    if 'MomentumOptimizer' in cols:
        cols.insert(0, cols.pop(cols.index('MomentumOptimizer')))
    if 'GradientDescentOptimizer' in cols:
        cols.insert(0, cols.pop(cols.index('GradientDescentOptimizer')))
    bm_table_pd = bm_table_pd.reindex(columns=cols)
    print(bm_table_pd)
    return bm_table_pd


def texify_plot_table(perf_table_pd, problem_set):
    """Write a ``.tex`` file with the performance table.

    The function will create a file named `performance_table_small.tex` or
    `performance_table_large.tex` with the latex code for the performance table.

    Args:
        perf_table_pd (pandas.dataframe): Pandas data frame for the performance
            table.
        problem_set (str): Can either be ``small`` or ``large`` to switch
            between which benchmark set is being plotted.

    Returns:
        str: String of the latex code for the performance table.

    """
    if not perf_table_pd.empty:
        # Postprocessing for Latex Output
        pd.set_option('display.max_colwidth', -1)
        perf_table_pd_n = perf_table_pd.apply(
            norm, axis=1)  # normalize between 0 and 100
        perf_table_pd_n_str = perf_table_pd_n.applymap(
            add_color_coding_tex) + perf_table_pd.applymap(
                latex)  # combine normalise version with latex color code command
        perf_table_pd_n_str.columns = perf_table_pd_n_str.columns.str.replace(
            '_', r'\_')  # Texify the column headers
        tikz_code = r"\def\cca#1#2{\cellcolor{green!#1!red}\ifnum #1<50\color{white}\fi{#2}}" +\
        "\n" + r"\resizebox{\textwidth}{!}{%" + "\n" +\
        perf_table_pd_n_str.to_latex(escape=False) + r"}"
        with open('performance_table_' + problem_set + '.tex', 'w') as tex_file:
            tex_file.write(tikz_code)

        return tikz_code


def norm(x):
    """Normalize the input of x, depending on the name (higher is better if
    test_acc is used, otherwise lower is better)"""
    if x.name[1] == 'Tuneability':
        return x
    if x.min() == x.max():
        return x - x.min() + 50.0
    if x.name[1] == 'Performance':
        if x.name[0] == "quadratic_deep" or x.name[0] == "mnist_vae" or x.name[
                0] == "fmnist_vae":
            return np.abs((x - x.max()) / (x.min() - x.max()) * 100)
        else:
            return np.abs((x - x.min()) / (x.max() - x.min()) * 100)
    else:
        return np.abs((x - x.max()) / (x.min() - x.max()) * 100)


def latex(input):
    """Create the latex output version of the input."""
    if isinstance(input, float):
        input = "%.4f" % input
        return "{" + str(input) + "}"
    elif isinstance(input, int):
        return "{" + str(input) + "}"
    elif isinstance(input, dict):
        return str(input).replace('{', '').replace('}', '').replace(
            "'", '').replace('_', '')
    else:
        return ""


def add_color_coding_tex(input):
    """Adds the latex command for color coding to the input"""
    if isinstance(input, str) or isinstance(input, int) or isinstance(
            input, float) and not np.isnan(input):
        return "\cca{" + str(int(input)) + "}"
    else:
        return ""
