# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from .tuner_utils import _dump_json, _load_json, _append_json, _clear_json


def plot_bo_posterior(optimizer_path, params, step):
    if type(params) == str:
        fig, ax = plot_1d_bo_posterior(optimizer_path, params, step)
    elif len(params) ==2:
        fig, ax = plot_2d_bo_posterior(optimizer_path, params, step)
    else:
        # TODO raise exception
        print('waasdsa')

def plot_2d_bo_posterior(optimizer_path, params, step):
    return 1,2

def plot_1d_bo_posterior(optimizer_path, param, step):
    # TODO do the steps really align?? check if posterior at each step is correct
    # TODO plot the evaluation points and the acq
    json_data = _load_json(optimizer_path, 'bo_plotting_data.json')
    
    domain = np.array(json_data['domain'][param])
    posterior_mean = np.array(json_data[str(step)]['posterior'][0])
    posterior_std = np.array(json_data[str(step)]['posterior'][1])
    acquisition = np.array(json_data[str(step)]['acquisition'])
    
#    domain_path = os.path.join(optimizer_path, 'domain.txt')
#    domain = np.loadtxt(domain_path)
    
#    posterior_mean_path = os.path.join(optimizer_path, 'posterior_mean_step_' + str(step) + '.txt')
#    posterior_mean = np.loadtxt(posterior_mean_path)
#    
#    posterior_std_path = os.path.join(optimizer_path, 'posterior_std_step_' + str(step) + '.txt')
#    posterior_std = np.loadtxt(posterior_std_path)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(domain, posterior_mean)
    ax[0].fill_between(domain, posterior_mean-posterior_std, posterior_mean+posterior_std, alpha=0.3)
    ax[1].plot(domain, acquisition)
    
    # add init and step points
    init_pairs, step_pairs = _read_eval_points_from_bo_tuning_summary(os.path.join(optimizer_path, 'bo_tuning_log.json'))
    # plot init pairs
    x = np.array([pair[0][param] for pair in init_pairs])
    y = np.array([pair[1]['target'] for pair in init_pairs])
    ax[0].scatter(x,y)
    # plot step pairs
    x = np.array([pair[0][param] for pair in step_pairs])
    y = np.array([pair[1]['target'] for pair in step_pairs])
    ax[0].scatter(x,y)
    
    plt.show()
    return fig, ax

def _generate_posterior_domain_from_bounds(bounds):
    keys = sorted(bounds)
    n_samples = 10000
    # make sure that the order in the array is the same as for the fitting in the optimization
    # the same step is done in the suggest() of the BO object
    domain = np.asarray([list(np.linspace(*bounds[key], n_samples)) for key in keys])
    return domain.T



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

def _init_bo_plotting_summary(utility_func, gp, bounds, log_path):
    # TODO how to properly sample? latin hypercube from prior?
    domain = _generate_posterior_domain_from_bounds(bounds)
    keys = sorted(bounds)
    
    domain_dict = dict(zip(keys, domain.T.tolist()))
#    np.savetxt(os.path.join(log_path, 'domain.txt'), domain)
    
    # init the json for the plotting data
    plotting_dict = {}
    plotting_dict['domain'] = domain_dict
    
    posterior_mean, posterior_std = gp.predict(domain, return_std = True)
    # TODO why y_max = 0 as last argument for utility?
    acquisition = utility_func.utility(domain, gp, 0)     
    # save plotting data
    plotting_dict[str(0)] = {'posterior': (list(posterior_mean), list(posterior_std)),
                                  'acquisition': list(acquisition)}
    _dump_json(log_path, 'bo_plotting_data.json', plotting_dict)
    
    # return domain array to calculate the posteroir later on
    return domain

def _update_bo_plotting_summary(utility_func, gp, iteration, domain, log_path):
    
    posterior_mean, posterior_std = gp.predict(domain, return_std = True)
    # TODO why y_max = 0 as last argument for utility?
    acquisition = utility_func.utility(domain, gp, 0)     
    # save plotting data
    json_data = _load_json(log_path, 'bo_plotting_data.json')
    json_data[str(iteration)] = {'posterior': (list(posterior_mean), list(posterior_std)),
                                  'acquisition': list(acquisition)}
    _dump_json(log_path, 'bo_plotting_data.json', json_data)
#    np.savetxt(os.path.join(log_path, 'posterior_mean_step_' + str(iteration) + '.txt'), posterior_mean)
#    np.savetxt(os.path.join(log_path, 'posterior_std_step_' + str(iteration) + '.txt'), posterior_std)
#    np.savetxt(os.path.join(log_path, 'acquisition_step_' + str(iteration) + '.txt'), acquisition)
    
def _update_bo_tuning_summary(op, iteration, log_path):
    
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
    summary_dict['predicted_target'] = predicted_target_mean[0]
    summary_dict['target'] = target
    _append_json(log_path, 'bo_tuning_log.json', (parameters, summary_dict))

def _read_eval_points_from_bo_tuning_summary(path_to_json):
    init_pairs = []
    step_pairs = []
    with open(path_to_json, 'r') as f:
        for line in f:
            line = json.loads(line)
            if not 'predicted_target' in line[1]:
                init_pairs.append(line)
            else:
                step_pairs.append(line)
    return init_pairs, step_pairs
            