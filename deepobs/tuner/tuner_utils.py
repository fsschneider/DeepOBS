# -*- coding: utf-8 -*-
from bayes_opt import UtilityFunction
import numpy as np
from bayes_opt import observer
import os
from bayes_opt.event import Events
import json
from scipy.stats. distributions import uniform
import matplotlib.pyplot as plt

# TODO all these analyses break if I start multiple tries with different params in the same results folder

# TODO somehow use the ranking to automate 10 seeds run?
def return_tuning_ranking(optimizer_path, metric = 'final_test_loss'):
    # TODO the ranking is flipped if I have accuracy
    # TODO maybe just set -loss for all analysis purposes to be consistent?
    generate_tuning_summary(optimizer_path)
    with open(os.path.join(optimizer_path, 'tuning_log.json'), 'r') as f:
        json_data = json.load(f)
    step_ranking = sorted(json_data, key = lambda step: json_data[step][metric])
    
    ranked_list = [{'parameters': json_data[step]['parameters'], metric: json_data[step][metric]} for step in step_ranking]
    
    return ranked_list

def read_in_parameter_performances(optimizer_path, hyperparams):
    if type(hyperparams) == str:
        hyperparams = hyperparams.split()
    
    with open(os.path.join(optimizer_path, 'tuning_log.json'), 'r') as f:
        json_data = json.load(f)
        
    steps = [step for step in json_data.keys() if 'Step' in step]
    
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
    
    optimizer_name = optimizer_path.split('/')[-1]
    testproblem = optimizer_path.split('/')[-2]
    
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

# TODO  the current setting that is in evaluation will not be listed
def generate_tuning_summary(optimizer_path):
    summary_dict = {}
    os.chdir(optimizer_path)
    setting_folders_ordered_by_creation_time = sorted(filter(os.path.isdir, os.listdir(optimizer_path)), key=os.path.getctime)
    step = 1
    for setting_folder in setting_folders_ordered_by_creation_time:
        # TODO How to do this in case there are several seed runs for one setting
        sub_dict = {}
        file_name = os.listdir(setting_folder)[0]
        path = os.path.join(optimizer_path, setting_folder, file_name)
        with open(path, 'r') as f:
            runner_output = json.load(f)
        # TODO step order does not matter for random and grid search but for bayesian
        sub_dict['parameters'] = runner_output['optimizer_hyperparams']
        # TODO final hard coded
        sub_dict['final_test_loss'] = runner_output['test_losses'][-1]
        # TODO will not work for tf version
        sub_dict['final_test_accuracy'] = runner_output['test_accuracies'][-1]
       
        summary_dict['Step_' + str(step)] = sub_dict
        step += 1
    
    with open(os.path.join(optimizer_path, 'tuning_log.json'), 'w') as f:
        f.write(json.dumps(summary_dict))

class log_uniform():        
    def __init__(self, a, b, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=1, random_state=None):
        uniform_values = uniform(loc=self.loc, scale=self.scale)
        return np.power(self.base, uniform_values.rvs(size=size, random_state=random_state))

class AdvancedObserver(observer._Tracker):
    def __init__(self, path, bounds, posterior_domain):
        self._path = path if path[-5:] == ".json" else path + ".json"
        try:
            os.remove(self._path)
        except OSError:
            pass

        # TODO how to deal with posterior domain for discrete variables?
        if posterior_domain is None:
            self._posterior_domain = self._generate_posterior_domain_from_bounds(bounds)
        else:
            self._posterior_domain = posterior_domain

        super(AdvancedObserver, self).__init__()

        self._init_json()

    def _init_json(self):
        with open(self._path, 'w') as f:
            dic = {}
#            dic['posterior_domain'] = self._posterior_domain
            f.write(json.dumps(dic))

    def _generate_posterior_domain_from_bounds(self, bounds):
        domain = {}
        for key, value in bounds.items():
            domain[key] = list(np.linspace(*value, 100))
        return domain

    def update(self, event, instance):

        if event == Events.OPTMIZATION_STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }

            # TODO calculate predicted target
#            predicted_target = self._get_predicted_target(instance)
            # TODO SIMPLIFY
            abstract_data = {}
            abstract_data['Step ' + str(self._iterations)] = data
            with open(self._path, 'r') as f:
                json_data = json.load(f)
            json_data.update(**abstract_data)
            with open(self._path, 'w') as f:
                f.write(json.dumps(json_data))

            # write utility and posterior to seperate files
            plotting_data = {}
            plotting_data['step'] = self._iterations
            mean, var = self._posterior(instance, self._posterior_domain)
            # TODO abstract domain to own file (same for every step)
            plotting_data['posterior_domain'] = self._posterior_domain
            plotting_data['posterior_mean'] = mean
            plotting_data['posterior_var'] = var
            # TODO understand utility arg 0
            utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
            utility = utility_function.utility(self._posterior_domain, instance._gp, 0)
            plotting_data['utility'] = list(utility)
            # TODO correct path
            with open('some_path_' + 'step_' + str(self._iterations), 'w') as f:
                f.write(json.dumps(plotting_data))

        self._update_tracker(event, instance)

    def _posterior(self, optimizer, domain):
        x_obs = []
        domain = []
        for param_name, param_domain in domain.items():
            x_obs.append([res["params"][param_name] for res in optimizer.res])
            domain.append(param_domain)
        x_obs = np.array(x_obs).T
        domain = np.array(domain).T

        y_obs = np.array([res["target"] for res in optimizer.res])

        optimizer._gp.fit(x_obs, y_obs)

        mu, sigma = optimizer._gp.predict(domain, return_std=True)
        return list(mu), list(sigma)