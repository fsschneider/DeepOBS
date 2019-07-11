# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats. distributions import uniform
from ..analyzer.shared_utils import create_setting_analyzer_ranking, _clear_json, _append_json, _determine_available_metric


def plot_2d_tuning_summary(optimizer_path, hyperparam, mode = 'final', xscale = 'linear', aggregated = False):
    # TODO
    return


def write_tuning_summary(optimizer_path, mode = 'final', metric = 'test_accuracies'):
    tuning_summary = generate_tuning_summary(optimizer_path, mode, metric)
    # clear json
    _clear_json(optimizer_path, 'tuning_log.json')
    for line in tuning_summary:
        _append_json(optimizer_path, 'tuning_log.json', line)


def generate_tuning_summary(optimizer_path, mode = 'final', metric = 'test_accuracies'):
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
        line = {'params': sett.aggregate['optimizer_hyperparams'], 'target_mean': target_mean, 'target_std': target_std}
        tuning_summary.append(line)
    return tuning_summary


class log_uniform():        
    def __init__(self, a, b, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=1, random_state=None):
        uniform_values = uniform(loc=self.loc, scale=self.scale)
        exp_values = np.power(self.base, uniform_values.rvs(size=size, random_state=random_state))
        if len(exp_values)==1:
            return exp_values[0]
        else:
            return exp_values
