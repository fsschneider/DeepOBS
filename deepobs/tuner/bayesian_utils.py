# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from .tuner_utils import _dump_json, _load_json, _append_json, _clear_json
from itertools import product

# TODO how to do slicing for d >2?
def _reshape_posterior_and_domain_for_plotting(mean, std, domain, acq, resolution):
    num_features = domain.shape[-1]
    new_shape = tuple(np.repeat(resolution, num_features))
    mean = mean.reshape(new_shape)
    std = std.reshape(new_shape)
    acq = acq.reshape(new_shape)
    new_domain = []
    for idx in range(num_features):
        new_domain.append(domain[:,idx].reshape(new_shape))
    return mean, std, new_domain, acq
    
def plot_bo_posterior(optimizer_path, step, resolution):
    # check whether the tuning run can be plotted (i.e. dim(GP)<=2)
    with open(os.path.join(optimizer_path, 'bo_tuning_log.json'), 'r') as f:
        first_line = json.loads(f.readline())
    dim = len(first_line[0])
    if dim == 1:
        fig, ax = plot_1d_bo_posterior(optimizer_path, step, resolution)
    elif dim ==2:
        fig, ax = plot_2d_bo_posterior(optimizer_path, step, resolution)
    else:
        raise NotImplementedError
        
def plot_2d_bo_posterior(optimizer_path, step, resolution):
    op = _load_bo_optimizer_object(os.path.join(optimizer_path, 'obj'), str(step))
    acq_func = _load_bo_optimizer_object(os.path.join(optimizer_path, 'obj'), 'acq_func')
    
    mean, std, domain = _calculate_posterior_from_op(op, resolution)
    acq = acq_func.utility(domain, op._gp, 0)
    
    mean, std, domain, acq = _reshape_posterior_and_domain_for_plotting(mean, std, domain, acq, resolution)
    
    fig, ax = plt.subplots(2,1)
    ax[0].contourf(domain[0], domain[1], mean)
    ax[0].set_xlabel(op.space.keys[0])
    ax[0].set_ylabel(op.space.keys[1])
    # TODO how to plot posterior std in 3d?
    
    # add step points
    ax[0].scatter(op.space.params[:,0], op.space.params[:,1])
    
    ax[1].contourf(domain[0], domain[1], acq)
    plt.show()
    return fig, ax

# TODO 1d 2d plot share abstraction
def plot_1d_bo_posterior(optimizer_path, step, resolution):
    op = _load_bo_optimizer_object(os.path.join(optimizer_path, 'obj'), str(step))
    acq_func = _load_bo_optimizer_object(os.path.join(optimizer_path, 'obj'), 'acq_func')
    
    mean, std, domain = _calculate_posterior_from_op(op, resolution)
    acq = acq_func.utility(domain, op._gp, 0)
    
    mean, std, domain, acq = _reshape_posterior_and_domain_for_plotting(mean, std, domain, acq, resolution)
    domain = np.squeeze(domain)
    acq = np.squeeze(acq)
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(domain, mean)
    ax[0].fill_between(domain, mean-std, mean+std, alpha=0.3)
    ax[0].set_xlabel(op.space.keys[0])
    ax[0].set_ylabel('target')
    ax[1].plot(domain, acq)
    
    # add step points
    ax[0].scatter(op.space.params, op.space.target)
    plt.show()
    return fig, ax

def _generate_domain_from_op(op, resolution):
    keys = op.space.keys
    bounds = op.space.bounds
    
    linspaces = []
    for idx, key in enumerate(keys):
        linspaces.append(np.linspace(*bounds[idx], num=resolution))
    
    domain = np.array(list(product(*linspaces)))
    return domain

def _calculate_posterior_from_op(op, resolution):
    domain = _generate_domain_from_op(op, resolution)
    mean, std = op._gp.predict(domain, return_std = True)
    return mean, std, domain

def _init_bo_tuning_summary(log_path, op):
    # clear json
    _clear_json(log_path, 'tuning_log.json')

def _save_bo_optimizer_object(path, file_name, op):
    with open(os.path.join(path, file_name), 'wb') as f:
        pickle.dump(op, f)
        
def _load_bo_optimizer_object(path, file_name):
    with open(os.path.join(path, file_name), 'rb') as f:
       op =  pickle.load(f)
    return op

def _update_bo_tuning_summary(gp, next_point, target, log_path):
    
    grid =  np.zeros((1, len(next_point)))
    c = 0
    for key, value in next_point.items():
        grid[0,c] = value
        c += 1
        
    predicted_target_mean, predicted_target_std = gp.predict(grid, return_std = True)
    
    summary_dict = {}
    summary_dict['predicted_target'] = predicted_target_mean[0]
    summary_dict['target'] = target
    _append_json(log_path, 'bo_tuning_log.json', (next_point, summary_dict))            