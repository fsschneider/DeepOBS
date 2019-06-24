# -*- coding: utf-8 -*-
import numpy as np
import os
import json
from scipy.stats. distributions import uniform
import matplotlib.pyplot as plt
import shutil

# TODO somehow use the ranking to automate 10 seeds run?

def plot_bo_posterior(optimizer_path, step):
    # TODO do the steps really align?? check if posterior at each step is correct
    # TODO how to distuingish 1d 2d and >2d not possible?
    # TODO plot the evaluation points and the acq
    json_data = _load_json(optimizer_path, 'tuning_log.json')
    
    domain_path = os.path.join(optimizer_path, 'domain.txt')
    domain = np.loadtxt(domain_path)
    
    posterior_mean_path = os.path.join(optimizer_path, 'posterior_mean_step_' + str(step) + '.txt')
    posterior_mean = np.loadtxt(posterior_mean_path)
    
    posterior_std_path = os.path.join(optimizer_path, 'posterior_std_step_' + str(step) + '.txt')
    posterior_std = np.loadtxt(posterior_std_path)
    
    plt.plot(domain, posterior_mean)
    plt.fill_between(domain, posterior_mean-posterior_std, posterior_mean+posterior_std, alpha=0.3)
    plt.show()
    
def create_tuning_ranking(optimizer_path, mode = 'final'):
    # TODO is bayes was run in different mode than it should not be possible to get the ranking according to wrong mode
    # make sure that summary is up to date
    generate_tuning_summary(optimizer_path)
    json_data = _load_json(optimizer_path, 'tuning_log.json')
    
    if mode + '_test_accuracy' in json_data['step_1']:
        sgn = -1
        metric = mode + '_test_accuracy'
    elif mode + '_test_loss' in json_data['step_1']:
        sgn = 1
        metric = mode + '_test_loss'
    else:
        # TODO raise exception
        print('something wrong')
        
    step_ranking = sorted(json_data, key = lambda step: sgn*json_data[step][metric])
    ranked_list = [{'parameters': json_data[step]['parameters'], metric: json_data[step][metric]} for step in step_ranking]
    return ranked_list

def read_in_parameter_performances(optimizer_path, hyperparams, mode = 'final'):
    # TODO create general API for all tuning methods and implement target by mode
    if type(hyperparams) == str:
        hyperparams = hyperparams.split()
    
    json_data = _load_json(optimizer_path, 'tuning_log.json')
        
    steps = [step for step in json_data.keys() if 'step' in step]
    
    list_dict = {param: [] for param in hyperparams}
    list_dict['final_test_loss'] = []
    for step in steps:
        print(step)
        for param in hyperparams:
            list_dict[param].append(json_data[step]['parameters'][param])
        list_dict['final_test_loss'].append(json_data[step]['final_test_loss'])
    return list_dict

def plot_1d_tuning_summary(optimizer_path, hyperparam, metric = 'final_test_loss', xscale = 'linear'):
    # make sure that summary is up to date
    generate_tuning_summary(optimizer_path)
    
    optimizer_name = os.path.split(optimizer_path)[-1]
    testproblem = os.path.split(optimizer_path)[-2]
    
    list_dict = read_in_parameter_performances(optimizer_path, hyperparam)
    # create array for plotting
    param_values = list_dict[hyperparam]
    metric_values = list_dict[metric]
    # sort the values synchronised for plotting
    param_values, metric_values = (list(t) for t in zip(*sorted(zip(param_values, metric_values))))
    fig, ax = plt.subplots()
    # TODO is it possible to determine the scale automatically by the distr, grid, etc?
    ax.plot(param_values, metric_values)
    plt.xscale(xscale)
    ax.set_title(optimizer_name + ' on ' + testproblem)
    plt.show()
    return fig, ax

def plot_2d_tuning_summary(optimizer_path, hyperparams, metric = 'final_test_loss'):
    # TODO how to resacle 3d plots x and t axis? (Imagine logubiforn combined with u niform distr.)
    list_dict = read_in_parameter_performances(optimizer_path, hyperparams)
    param_values1 = list_dict[hyperparams[0]]
    param_values2 = list_dict[hyperparams[1]]
    metric_values = list_dict[metric]
    plt.scatter(x=param_values1, y=param_values2, c=metric_values)
    # TODO normalize the heatmap because otherwise it is hard to distuingish between better runs
    plt.colorbar()
    plt.show()    
    return 4,3
#    return fig, ax

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
        # TODO raise exception
        print('metric not valid')
    
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
            aggregate[metrics] = {
                'mean': np.mean(eval(metrics), axis=0),
                'std': np.std(eval(metrics), axis=0)
            }
    # merge meta data
    aggregate['optimizer_hyperparams'] = json_data['optimizer_hyperparams']
    return aggregate

def get_aggregated_setting_summary(setting_path):
    # TODO aggregated summary needs to return params
    summary_dict = {}
    aggregate = aggregate_runs(setting_path)
    params = aggregate['optimizer_hyperparams']
    if 'test_accuracies' in aggregate:
        summary_dict['final_test_accuracy'] = aggregate['test_accuracies']['mean'][-1]
        summary_dict['best_test_accuracy'] = max(aggregate['test_accuracies']['mean'])
    summary_dict['final_test_loss'] = aggregate['test_losses']['mean'][-1]
    summary_dict['best_test_loss'] = min(aggregate['test_losses']['mean'])
    return (params, summary_dict)

def get_setting_file_summary(setting_path, json_file):
    json_data = _load_json(setting_path, json_file)
    parameters = json_data['optimizer_hyperparams']
    summary_dict = {}
    if 'test_accuracies' in json_data:
        summary_dict['final_test_accuracy'] = json_data['test_accuracies'][-1]
        summary_dict['best_test_accuracy'] = max(json_data['test_accuracies'])
    summary_dict['final_test_loss'] = json_data['test_losses'][-1]
    summary_dict['best_test_loss'] = min(json_data['test_losses'])
    return (parameters, summary_dict)

def generate_tuning_summary(optimizer_path, aggregated = False):
    # TODO make it clear that this is not used for bayesian
    
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
        return np.power(self.base, uniform_values.rvs(size=size, random_state=random_state))
    
def _generate_posterior_domain_from_bounds(bounds):
    keys = sorted(bounds)
    n_samples = 10000
    # make sure that the order in the array is the same as for the fitting in the optimization
    # the same step is done in the suggest() of the BO object
    domain = np.asarray([list(np.linspace(*bounds[key], n_samples)) for key in keys])
    return domain.T

def _append_json(path, file, obj):
    with open(os.path.join(path, file), 'a') as f:
        f.write(json.dumps(obj))
        f.write('\n')

def _clear_json(path, file):
    json_path = os.path.join(path, file)
    if os.path.exists(json_path):
        os.remove(json_path)

def _init_bo_tuning_summary(log_path, op):
    # adds init values if availble
    # clear json
    _clear_json(log_path, 'tuning_log.json')
    
    summary_dict = {}
    for idx, res in enumerate(op.res):
        params = res['params']
        summary_dict['target'] = res['target']
        _append_json(log_path, 'bo_tuning_log.json', (params, summary_dict))
    # TODO append a visual (and useful) guidance for end on init values

def _init_bo_plotting_summary(bounds, log_path):
    # TODO how to properly sample? latin hypercube from prior?
    # TODO it would be better to save the domain as dict to know which axis is which param
    domain = _generate_posterior_domain_from_bounds(bounds)
    np.savetxt(os.path.join(log_path, 'domain.txt'), domain)
    return domain

def _dump_json(path, file, obj):
    with open(os.path.join(path, file), 'w') as f:
        f.write(json.dumps(obj))

def _update_bo_plotting_summary(utility_func, gp, iteration, domain, log_path):
    posterior_mean, posterior_std = gp.predict(domain, return_std = True)
    # TODO dump all this in one json?
    # TODO why 0 as last argument for utility?
    acquisition = utility_func.utility(domain, gp, 0)     
    # save plotting data
    np.savetxt(os.path.join(log_path, 'posterior_mean_step_' + str(iteration) + '.txt'), posterior_mean)
    np.savetxt(os.path.join(log_path, 'posterior_std_step_' + str(iteration) + '.txt'), posterior_std)
    np.savetxt(os.path.join(log_path, 'acquisition_step_' + str(iteration) + '.txt'), acquisition)
    
def _update_bo_tuning_summary(op, utility_func, iteration, log_path):
    
    last_point = op.res[-1]['params']
    grid =  np.zeros((1, len(last_point)))
    c = 0
    for key, value in last_point.items():
        grid[0,c] = value
        c += 1
    target = op.res[-1]['target']
    predicted_target_mean, predicted_target_std = op._gp.predict(grid, return_std = True)
    
    parameters = last_point
    summary_dict = {}
    summary_dict['predicted_target'] = list(predicted_target_mean)
    summary_dict['target'] = target
    _append_json(log_path, 'bo_tuning_log.json', (parameters, summary_dict))