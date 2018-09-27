# -*- coding: utf-8 -*-
"""
This module contains utility functions for plotting.
"""

from __future__ import print_function

import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

import deepobs


class optimizer_run:
    def __init__(self, file, name=None):
        with (open(file, "rb")) as openfile:
            pickle_file = pickle.load(openfile)
            args = vars(pickle_file['args'])
            # Name and Testproblem
            if name is not None:
                self.name = name
            else:
                self.name = str(file).split('.')[-2].split('/')[-1]
            self.test_problem = args['test_problem']
            # Checkpoint
            self.train_loss_mean = pickle_file['checkpoint/checkpoint_train_loss']
            self.test_loss_mean = pickle_file['checkpoint/checkpoint_test_loss']
            self.train_acc_mean = pickle_file['checkpoint/checkpoint_train_acc']
            self.test_acc_mean = pickle_file['checkpoint/checkpoint_test_acc']
            self.steps = pickle_file['checkpoint/checkpoint_steps']
            # Training
            self.training_steps = pickle_file['training/training_steps']
            self.training_loss = pickle_file['training/training_loss']
            # Args for Run and Optimizer
            run_keys = ["num_epochs", "wd", "data_dir", "nologs", "bs", "saveto", "train_log_interval", "checkpoint_epochs", "print_train_iter", "no_time", "pickle", "random_seed", "test_problem", "run_name"]
            self.opt_args = args
            self.run_args = dict([(k, self.opt_args.pop(k)) for k in run_keys])
            # Remove arguments with None or False from opt_args
            self.opt_args = {key: val for key, val in self.opt_args.items() if val is not None and val is not False}

            # Set Standard deviation to zero, must be added from outside
            self.train_loss_std = np.zeros_like(self.train_loss_mean)
            self.test_loss_std = np.zeros_like(self.test_loss_mean)
            self.train_acc_std = np.zeros_like(self.train_acc_mean)
            self.test_acc_std = np.zeros_like(self.test_acc_mean)

            # Performance measures
            conv_perf, use_acc = deepobs.run_utils.convergence_performance(self)
            if use_acc:
                self.final_performance = self.test_acc_mean[-1]
            else:
                self.final_performance = self.test_loss_mean[-1]
            if 'time/convergence_iterations' in pickle_file and np.max(pickle_file['time/convergence_iterations']) != 0:
                self.speed = pickle_file['time/convergence_iterations'][-1]
            else:
                self.speed = self.run_args['num_epochs'] + 1

            # Number of Runs (1 in general, but >1 if std is computed)
            self.num_runs = 1

    def plot(self, ax, lc=None, lw=None, ls=None):
        for idx, measure in enumerate([(self.test_loss_mean, self.test_loss_std), (self.train_loss_mean, self.train_loss_std), (self.test_acc_mean, self.test_acc_std), (self.train_acc_mean, self.train_acc_std)]):
            ax[idx].plot(self.steps, measure[0], label=self.name, lw=lw, ls=ls, color=lc)
            ax[idx].fill_between(self.steps, np.clip(measure[0] - measure[1], 1e-12, None), measure[0] + measure[1], color=ax[idx].get_lines()[-1].get_color(), alpha=0.2)


def get_filestructure(results_dir):
    # Read out all files and built a structure
    file_structure = dict()
    for test_problem in os.listdir(results_dir):
        if test_problem.startswith('.'):
            pass
        else:
            file_structure[test_problem] = {}
            for optimizer in os.listdir(os.path.join(results_dir, test_problem)):
                file_structure[test_problem][optimizer] = []
                for root, dirs, files in os.walk(os.path.join(results_dir, test_problem, optimizer)):
                    for file in files:
                        if file.endswith('.pickle'):
                            file_structure[test_problem][optimizer].append(os.path.join(root, file))
    return file_structure


def get_test_problem(file):
    with (open(file, "rb")) as openfile:
        pickle_file = pickle.load(openfile)
        args = vars(pickle_file['args'])
        test_problem = args['test_problem']
    return test_problem


def get_baselines(test_problem):
    test_data, test_model = test_problem.split('.')
    this_dir, this_filename = os.path.split(__file__)

    baselines_path = os.path.join(this_dir, test_data, "baselines", test_model)
    baselines = dict()
    for root, dirs, files in os.walk(baselines_path):
        files_with_pickle = filter(lambda s: s[-7:] == '.pickle', files)
        if root.endswith("Adam"):
            baselines["Adam"] = get_average_run([os.path.join(root, f) for f in files_with_pickle], name="Adam")
        elif root.endswith("Momentum"):
            baselines["Momentum"] = get_average_run([os.path.join(root, f) for f in files_with_pickle], name="Momentum")
        elif root.endswith("SGD"):
            baselines["SGD"] = get_average_run([os.path.join(root, f) for f in files_with_pickle], name="SGD")
    return baselines


def create_figure(name):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(4.5, 6), dpi=200)
    name = str(name).capitalize()
    f.canvas.set_window_title(name)
    f.suptitle(name)
    ax1.set_ylabel('Train Loss')
    ax2.set_ylabel('Test Loss')
    ax3.set_ylabel('Train Accuracy')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_xlabel('Epochs')
    return f, (ax1, ax2, ax3, ax4)


def get_average_run(runs, name):
    list_optimizer_runs = []
    # Loop over all runs (could be just one) and get optimizer_run instance from them, collect them in a list
    for run in runs:
        list_optimizer_runs.append(optimizer_run(run, name=name))
    # create new optimizer_run that holds the mean and std
    avg_optimizer_run = list_optimizer_runs[0]
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    final_performance = []
    speed = []
    for opt_run in list_optimizer_runs:
        train_loss.append(opt_run.train_loss_mean)
        test_loss.append(opt_run.test_loss_mean)
        train_acc.append(opt_run.train_acc_mean)
        test_acc.append(opt_run.test_acc_mean)
        final_performance.append(opt_run.final_performance)
        speed.append(opt_run.speed)
    # Compute statistics
    avg_optimizer_run.train_loss_mean, avg_optimizer_run.train_loss_std = np.mean(train_loss, axis=0), np.std(train_loss, axis=0)
    avg_optimizer_run.test_loss_mean, avg_optimizer_run.test_loss_std = np.mean(test_loss, axis=0), np.std(test_loss, axis=0)
    avg_optimizer_run.train_acc_mean, avg_optimizer_run.train_acc_std = np.mean(train_acc, axis=0), np.std(train_acc, axis=0)
    avg_optimizer_run.test_acc_mean, avg_optimizer_run.test_acc_std = np.mean(test_acc, axis=0), np.std(test_acc, axis=0)
    avg_optimizer_run.final_performance = np.mean(final_performance)
    avg_optimizer_run.speed = np.mean(speed)

    # change num_runs
    avg_optimizer_run.num_runs = len(list_optimizer_runs)
    avg_optimizer_run.name += "_avg_" + str(avg_optimizer_run.num_runs) + "_runs"

    return avg_optimizer_run


def set_figure(ax, log):
    ax[0].legend(fontsize='xx-small')
    ax[0].margins(x=0)
    ax[1].margins(x=0)
    ax[2].margins(x=0)
    ax[3].margins(x=0)
    if log:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[2].set_yscale('log')
        ax[3].set_yscale('log')
    # plt.tight_layout()


def add_color_coding_tex(input):
    """Adds the latex command for color coding to the input"""
    if isinstance(input, str) or isinstance(input, int) or isinstance(input, float):
        return "\cca{" + str(int(input)) + "}"
    else:
        return ""


def latex(input):
    """Create the latex output version of the input."""
    if isinstance(input, float):
        input = "%.2f" % input
        return "{" + str(input) + "}"
    elif isinstance(input, int):
        return "{" + str(input) + "}"
    elif isinstance(input, dict):
        return str(input).replace('{', '').replace('}', '').replace("'", '').replace('_', '')
    else:
        return ""


def norm(x):
    """Normalize the input of x, depending on the name (higher is better if test_acc is used, otherwise lower is better)"""
    if x.name[1] == 'Tuneability':
        return x
    if x.min() == x.max():
        return x - x.min() + 50.0
    if x.name[1] == 'Performance' and x.name[0] == texify("mnist.mnist_mlp") or x.name[0] == texify("fmnist.fmnist_2c2d" or x.name[0] == texify("cifar10.cifar10_3c3d") or x.name[0] == texify("cifar100.cifar100_allcnnc") or x.name[0] == texify("svhn.svhn_wrn164") or x.name[0] == texify("tolstoi.tolstoi_char_rnn")):
        return np.abs((x - x.min()) / (x.max() - x.min()) * 100)
    else:
        return np.abs((x - x.max()) / (x.min() - x.max()) * 100)


def texify(x):
    """Make the input x ready for latex, replace things that do not look good in Latex"""
    return str(x).replace('{', '').replace('}', '').replace("'", '').replace('_', '-')
