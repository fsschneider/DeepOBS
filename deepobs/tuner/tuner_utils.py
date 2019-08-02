# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats. distributions import uniform
from ..analyzer.shared_utils import create_setting_analyzer_ranking, _clear_json, _append_json, _determine_available_metric
import os


def rerun_setting(runner,
                  optimizer_class,
                  hyperparam_names,
                  optimizer_path,
                  seeds=np.arange(43, 52),
                  rank=1, mode='final',
                  metric='valid_accuracies'):
    """Reruns a hyperparameter setting with several seeds after the tuning is finished. Defaults to rerun the best setting.
    Args:
        runner (framework runner.runner): The runner which was used for the tuning.
        optimizer_class (framework optimizer class): The optimizer class that was tuned.
        hyperparam_names (dict): A nested dictionary that holds the names, the types and the default values of the hyperparams.
        optimizer_path (str): The path to the optimizer to analyse the best setting on.
        seeds (iterable): The seeds that are used to rerun the setting.
        rank (int): The ranking of the setting that is to rerun.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
    """
    metric = _determine_available_metric(optimizer_path, metric)
    optimizer_path = os.path.join(optimizer_path)

    setting_analyzer_ranking = create_setting_analyzer_ranking(optimizer_path, mode, metric)
    setting = setting_analyzer_ranking[rank - 1]

    runner = runner(optimizer_class, hyperparam_names)

    hyperparams = setting.aggregate['optimizer_hyperparams']
    training_params = setting.aggregate['training_params']
    testproblem = setting.aggregate['testproblem']
    num_epochs = setting.aggregate['num_epochs']
    batch_size = setting.aggregate['batch_size']
    results_path = os.path.split(os.path.split(optimizer_path)[0])[0]
    for seed in seeds:
        runner.run(testproblem, hyperparams=hyperparams, random_seed=int(seed), num_epochs=num_epochs,
                   batch_size=batch_size, output_dir=results_path, **training_params)


def write_tuning_summary(optimizer_path, mode = 'final', metric = 'valid_accuracies'):
    """Writes the tuning summary to a json file in the ``optimizer_path``.
    Args:
        optimizer_path (str): Path to the optimizer folder.
        mode (str): The mode on which the performance measure for the summary is based.
        metric (str): The metric which is printed to the tuning summary as 'target'
    """
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)

    # clear json
    _clear_json(optimizer_path, 'tuning_log.json')
    for line in tuning_summary:
        _append_json(optimizer_path, 'tuning_log.json', line)


def generate_tuning_summary(optimizer_path, mode = 'final', metric = 'valid_accuracies'):
    """Generates a list of dictionaries that holds an overview of the current tuning process.
    Should not be used for Bayesian tuning methods, since the order of evaluation is ignored in this summary. For
    Bayesian tuning methods use the tuning summary logging of the respective class.

    Args:
        optimizer_path (str): Path to the optimizer folder.
        mode (str): The mode on which the performance measure for the summary is based.
        metric (str): The metric which is printed to the tuning summary as 'target'
    Returns:
        tuning_summary (list): A list of dictionaries. Each dictionary corresponds to one hyperparameter evaluation
        of the tuning process and holds the hyperparameters and their performance.
        setting_analyzer_ranking (list): A ranked list of SettingAnalyzers that were used to generate the summary
        """
    metric = _determine_available_metric(optimizer_path, metric)
    setting_analyzer_ranking = create_setting_analyzer_ranking(optimizer_path, mode, metric)
    tuning_summary = []
    for sett in setting_analyzer_ranking:
        if mode == 'final':
            target_mean = sett.aggregate[metric]['mean'][-1]
            target_std = sett.aggregate[metric]['std'][-1]
        elif mode == 'best':
            idx = np.argmax(sett.aggregate[metric]['mean'])
            target_mean = sett.aggregate[metric]['mean'][idx]
            target_std = sett.aggregate[metric]['std'][idx]
        else:
            raise RuntimeError('Mode not implemented.')
        line = {'params': {**sett.aggregate['optimizer_hyperparams'], **sett.aggregate['training_params']}, metric + "_mean": target_mean, metric + '_std': target_std}
        tuning_summary.append(line)
    return tuning_summary


class log_uniform():
    """A log uniform distribution that takes an arbitrary base."""
    def __init__(self, a, b, base=10):
        """
        Args:
            a (float): Lower bound.
            b (float): Range from lower bound.
            base (float): Base of the log.
        """
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=1, random_state=None):
        uniform_values = uniform(loc=self.loc, scale=self.scale)
        exp_values = np.power(self.base, uniform_values.rvs(size=size, random_state=random_state))
        if len(exp_values) == 1:
            return exp_values[0]
        else:
            return exp_values
