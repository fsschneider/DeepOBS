# -*- coding: utf-8 -*-
from bayes_opt import UtilityFunction
import numpy as np
from bayes_opt import observer
import os
from bayes_opt.event import Events
import json
from scipy.stats. distributions import uniform
import matplotlib.pyplot as plt
from bayes_opt.util import UtilityFunction

# TODO all these analyses break if I start multiple tries with different params in the same results folder
# TODO somehow use the ranking to automate 10 seeds run?

def create_tuning_ranking(optimizer_path, mode = 'final'):
    # make sure that summary is up to date
    generate_tuning_summary(optimizer_path)
    json_data = _load_json(optimizer_path, 'tuning_log.json')
    
    if mode + '_test_accuracy' in json_data['Step_1']:
        sgn = -1
        metric = mode + '_test_accuracy'
    elif mode + '_test_loss' in json_data['Step_1']:
        sgn = 1
        metric = mode + '_test_loss'
    else:
        # TODO raise exception
        print('something wrong')
        
    step_ranking = sorted(json_data, key = lambda step: sgn*json_data[step][metric])
    ranked_list = [{'parameters': json_data[step]['parameters'], metric: json_data[step][metric]} for step in step_ranking]
    return ranked_list

def read_in_parameter_performances(optimizer_path, hyperparams):
    if type(hyperparams) == str:
        hyperparams = hyperparams.split()
    
    json_data = _load_json(optimizer_path, 'tuning_log.json')
        
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

    return aggregate

# TODO implement the summary for bayesian with advanced observer (because order matters) really? because DURING tuning it should be fine
def generate_tuning_summary(optimizer_path, mode = 'final'):
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
    def __init__(self, path, bounds, posterior_domain, acq_type, acq_kappa, acq_xi):
        self._path = os.path.join(path) if path[-5:] == ".json" else os.path.join(path, ".json")
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
        
        # needed for target prediction
        self._acq_type = acq_type
        self._acq_kappa = acq_kappa
        self._acq_xi = acq_xi
        
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
    
    def _get_predicted_target(self, instance):
        # save the random state since suggesting changes it
        random_state = instance._random_state
        utility = UtilityFunction(self._acq_type, kappa = self._acq_kappa, xi = self._acq_xi)
        suggested_point = instance.suggest(utility)
        instance._
        instance._random_state = random_state
        
        
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
            predicted_target = self._get_predicted_target(instance)
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
            mean, var = self._get_posterior(instance, self._posterior_domain)
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

    def _get_posterior(self, instance, domain):
        x_obs = []
        domain = []
        for param_name, param_domain in domain.items():
            x_obs.append([res["params"][param_name] for res in instance.res])
            domain.append(param_domain)
        x_obs = np.array(x_obs).T
        domain = np.array(domain).T

        y_obs = np.array([res["target"] for res in instance.res])

        instance._gp.fit(x_obs, y_obs)

        mu, sigma = instance._gp.predict(domain, return_std=True)
        return list(mu), list(sigma)
    
def _generate_posterior_domain_from_bounds(bounds):
        n_samples = 100
        domain = np.zeros((n_samples, len(bounds)))
        c=0
        for key, value in bounds.items():
            domain[:,c] = list(np.linspace(*value, n_samples))
            c+=1 
        return domain
    
def _init_bo_tuning_summary(log_path):
    json_dict = {}
    _dump_json(log_path, 'tuning_log.json', json_dict)
        
def _init_bo_plotting_summary(bounds, log_path):
    # TODO how to properly sample? latin hypercube from prior?
    # TODO it would be better to save the domain as dict to know which axis is which param
    domain = _generate_posterior_domain_from_bounds(bounds)
    np.savetxt(os.path.join(log_path, 'domain.txt'), domain)
    return domain

def _dump_json(path, file_name, json_data):
    with open(os.path.join(path, file_name), 'w') as f:
        f.write(json.dumps(json_data))

def _update_bo_plotting_summary(utility_func, gp, iteration, domain, log_path):
    posterior_mean, posterior_std = gp.predict(domain, return_std = True)
    # TODO why 0 as last argument for utility?
    acquisition = utility_func.utility(domain, gp, 0)     
    # save plotting data
    np.savetxt(os.path.join(log_path, 'posterior_mean_step_' + str(iteration) + '.txt'), posterior_mean)
    np.savetxt(os.path.join(log_path, 'posterior_std_step_' + str(iteration) + '.txt'), posterior_std)
    np.savetxt(os.path.join(log_path, 'acquisition_step_' + str(iteration) + '.txt'), acquisition)
    
def _update_bo_tuning_summary(op, utility_func, iteration, log_path):
    
    last_point = op.res[-1]['params']
    # TODO how the fuck does the gp know the order of the params?
    grid =  np.zeros((1, len(last_point)))
    c = 0
    for key, value in last_point.items():
        grid[0,c] = value
        c += 1
    actual_target = op.res[-1]['target']
    predicted_target_mean, predicted_target_std = op._gp.predict(grid, return_std = True)
    json_data = _load_json(log_path, 'tuning_log.json')
    json_data['Step_' + str(iteration)] = {'parameters': last_point,
              'predicted_target': list(predicted_target_mean),
              'predicted_target_std': list(predicted_target_std),
              'actual_target': actual_target
            }
    _dump_json(log_path, 'tuning_log.json', json_data)
    