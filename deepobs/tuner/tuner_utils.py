# -*- coding: utf-8 -*-
import numpy as np
import os
import json
from scipy.stats. distributions import uniform
import matplotlib.pyplot as plt


# TODO somehow use the ranking to automate 10 seeds run?


def _read_eval_pairs_from_tuning_summary(path_to_json):
    step_pairs = []
    with open(path_to_json, 'r') as f:
        for line in f:
            line = json.loads(line)
            step_pairs.append(line)
    return step_pairs


def create_tuning_ranking(optimizer_path, mode = 'final', aggregated = False):
    # TODO is bayes was run in different mode than it should not be possible to get the ranking according to wrong mode
    
    # make sure that summary is up to date
    generate_tuning_summary(optimizer_path, aggregated=aggregated)
    
    # read in the whole summary    
    eval_pairs = _read_eval_pairs_from_tuning_summary(os.path.join(optimizer_path, 'tuning_log.json'))
    
    # determine the sign for sorting
    if mode + '_test_accuracy' in eval_pairs[0][1]:
        sgn = -1
        metric = mode + '_test_accuracy'
    elif mode + '_test_loss' in eval_pairs[0][1]:
        sgn = 1
        metric = mode + '_test_loss'
    else:
        raise NotImplementedError
    
    if aggregated:
        step_ranking = sorted(eval_pairs, key = lambda step: sgn * step[1][metric]['mean'])
    else:
        step_ranking = sorted(eval_pairs, key = lambda step: sgn * step[1][metric])
    ranked_list = [{'parameters': step[0], metric: step[1][metric]} for step in step_ranking]
    return ranked_list


def plot_1d_tuning_summary(optimizer_path, hyperparam, mode = 'final', xscale = 'linear', aggregated = False):
    # make sure that summary is up to date
    generate_tuning_summary(optimizer_path, aggregated=aggregated)
    
    optimizer_name = os.path.split(optimizer_path)[-1]
    testproblem = os.path.split(optimizer_path)[-2]
    
    eval_pairs = _read_eval_pairs_from_tuning_summary(os.path.join(optimizer_path, 'tuning_log.json'))
    
    # determine metric from mode
    if mode + '_test_accuracy' in eval_pairs[0][1]:
        metric = mode + '_test_accuracy'
    elif mode + '_test_loss' in eval_pairs[0][1]:
        metric = mode + '_test_loss'
    else:
        raise NotImplementedError
        
    # create array for plotting
    param_values = [step[0][hyperparam] for step in eval_pairs]
    metric_values = [step[1][metric] for step in eval_pairs]
    # sort the values synchronised for plotting
    param_values, metric_values = (list(t) for t in zip(*sorted(zip(param_values, metric_values))))
    fig, ax = plt.subplots()
    # TODO is it possible to determine the scale automatically by the distr, grid, etc?
    if aggregated:
        metric_mean = np.array([value['mean'] for value in metric_values])
        metric_std = np.array([value['std'] for value in metric_values])
        ax.plot(param_values, metric_mean)
        ax.fill_between(param_values, metric_mean-metric_std, metric_mean + metric_std, alpha = 0.3)
    else:
        ax.plot(param_values, metric_values)
    plt.xscale(xscale)
    ax.set_title(optimizer_name + ' on ' + testproblem)
    plt.show()
    return fig, ax


def plot_2d_tuning_summary(optimizer_path, hyperparam, mode = 'final', xscale = 'linear', aggregated = False):
    # TODO
    return


def get_aggregated_setting_summary(setting_path):
    summary_dict = {}
    aggregate = aggregate_runs(setting_path)
    params = aggregate['optimizer_hyperparams']
    if 'test_accuracies' in aggregate:
        summary_dict['final_test_accuracy'] = {'mean': aggregate['test_accuracies']['mean'][-1],
                                                'std': aggregate['test_accuracies']['std'][-1]}
        
        idx = np.argmax(aggregate['test_accuracies']['mean'])
        summary_dict['best_test_accuracy'] = {'mean': aggregate['test_accuracies']['mean'][idx],
                                                'std': aggregate['test_accuracies']['std'][idx]}
        
    summary_dict['final_test_loss'] = {'mean': aggregate['test_losses']['mean'][-1],
                                                'std': aggregate['test_losses']['std'][-1]}
    
    idx = np.argmin(aggregate['test_losses']['mean'])
    summary_dict['best_test_loss'] = {'mean': aggregate['test_losses']['mean'][idx],
                                    'std': aggregate['test_losses']['std'][idx]}
    return params, summary_dict


def get_setting_file_summary(setting_path, json_file):
    json_data = _load_json(setting_path, json_file)
    parameters = json_data['optimizer_hyperparams']
    summary_dict = {}
    if 'test_accuracies' in json_data:
        summary_dict['final_test_accuracy'] = json_data['test_accuracies'][-1]
        summary_dict['best_test_accuracy'] = max(json_data['test_accuracies'])
    summary_dict['final_test_loss'] = json_data['test_losses'][-1]
    summary_dict['best_test_loss'] = min(json_data['test_losses'])
    return parameters, summary_dict


def generate_tuning_summary(optimizer_path, aggregated = False):
    os.chdir(optimizer_path)
    setting_folders = [d for d in os.listdir(optimizer_path) if os.path.isdir(os.path.join(optimizer_path,d))]
    
    # clear json
    _clear_json(optimizer_path, 'tuning_log.json')
    
    for setting_folder in setting_folders:
        if aggregated:
            path = os.path.join(optimizer_path, setting_folder)
            summary_tuple = get_aggregated_setting_summary(path)    
        else:
            file_name = os.listdir(setting_folder)[0]
            path = os.path.join(optimizer_path, setting_folder)
            summary_tuple = get_setting_file_summary(path, file_name)
        
        # append to the json
        _append_json(optimizer_path, 'tuning_log.json', summary_tuple)


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


def _dump_json(path, file, obj):
    with open(os.path.join(path, file), 'w') as f:
        f.write(json.dumps(obj))


def _append_json(path, file, obj):
    with open(os.path.join(path, file), 'a') as f:
        f.write(json.dumps(obj))
        f.write('\n')


def _clear_json(path, file):
    json_path = os.path.join(path, file)
    if os.path.exists(json_path):
        os.remove(json_path)


def _load_json(path, file_name):
    with open(os.path.join(path, file_name), "r") as f:
         json_data = json.load(f)
    return json_data


def compute_speed(setting_folder, conv_perf, metric):
    runs = [run for run in os.listdir(setting_folder) if run.endswith(".json")]
    # metrices
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for run in runs:
        json_data = _load_json(setting_folder, run)
        train_losses.append(json_data['train_losses'])
        test_losses.append(json_data['test_losses'])
        # just add accuracies to the aggregate if they are available
        if 'train_accuracies' in json_data :
            train_accuracies.append(json_data['train_accuracies'])
            test_accuracies.append(json_data['test_accuracies'])
    
    perf = np.array(eval(metric))
    if metric == "test_losses" or metric == "train_losses":
        # average over first time they reach conv perf (use num_epochs if conv perf is not reached)
        speed = np.mean(
            np.argmax(perf <= conv_perf, axis=1) +
            np.invert(np.max(perf <= conv_perf, axis=1)) *
            perf.shape[1])
    elif metric == "test_accuracies" or metric == "train_accuracies":
        speed = np.mean(
            np.argmax(perf >= conv_perf, axis=1) +
            np.invert(np.max(perf >= conv_perf, axis=1)) *
            perf.shape[1])
    else:
        raise NotImplementedError
    
    return speed


def aggregate_runs(setting_folder):
    runs = [run for run in os.listdir(setting_folder) if run.endswith(".json")]
    # metrices
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for run in runs:
        json_data = _load_json(setting_folder, run)
        train_losses.append(json_data['train_losses'])
        test_losses.append(json_data['test_losses'])
        # just add accuracies to the aggregate if they are available
        if 'train_accuracies' in json_data :
            train_accuracies.append(json_data['train_accuracies'])
            test_accuracies.append(json_data['test_accuracies'])

    aggregate = dict()
    for metrics in ['train_losses', 'test_losses', 'train_accuracies', 'test_accuracies']:
        # only add the metric if available
        if len(eval(metrics)) != 0:
        # TODO exclude the parts which are NaN?
            try:
                aggregate[metrics] = {
                        'mean': np.mean(eval(metrics), axis=0),
                        'std': np.std(eval(metrics), axis=0)
                    }
            except:
                pass
    # merge meta data
    aggregate['optimizer_hyperparams'] = json_data['optimizer_hyperparams']
    return aggregate
