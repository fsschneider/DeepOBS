# -*- coding: utf-8 -*-
"""
This module contains utility functions to be used accross multiple run scripts.
"""

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle
import os


def make_learning_rate_schedule(args):
    """Produce the learning rate tensor from the command line arguments of a run script.

    Currently implements constant learning rates (using the --lr command line
    argument) and step-wise learning rate schedules (using the --lr_sched_epochs
    and --_lr_sched_factors arguments)."""
    assert args.lr is not None
    if args.lr_sched_epochs is None or args.lr_sched_factors is None:
        assert args.lr_sched_epochs is None and args.lr_sched_factors is None
        learning_rate = np.ones(args.num_epochs) * args.lr
    else:
        assert len(args.lr_sched_factors) == len(args.lr_sched_epochs)
        learning_rate = np.ones(args.num_epochs) * args.lr
        for idx, stp in enumerate(args.lr_sched_epochs):
            learning_rate[stp:] = args.lr * args.lr_sched_factors[idx]
    return learning_rate


def float2str(x):
    """Helper function to convert floats to standard-format strings."""
    a = '%E' % x
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def dump2pickle(args, summary_writer, name):
    event_acc = EventAccumulator(summary_writer.get_logdir())
    event_acc.Reload()
    results = {"args": args}
    # Show all tags in the log file
    for s in event_acc.Tags()['scalars']:
        _, steps, vals = zip(*event_acc.Scalars(s))
        results[s] = list(vals)
        if s == 'checkpoint/checkpoint_test_loss':
            results['checkpoint/checkpoint_steps'] = map(int, list(steps))
        elif s == 'training/training_loss':
            results['training/training_steps'] = map(int, list(steps))

    pickle_name = name.replace("/", "__")
    with open(os.path.join(summary_writer.get_logdir(), "results__" + pickle_name + ".pickle"), "w") as f:
        pickle.dump(results, f)


def convergence_performance(args):
    """Returns the convergence performances for a test problem. The convergence performance is defined as the minimum test accuracy an optimizer needs to reach before we say it convergenced on this test problem."""
    # Dict with convergence performances using test accuracy
    conv_perf = {"mnist.mnist_mlp": [0.97], "fmnist.fmnist_2c2d": [0.9186], "cifar10.cifar10_3c3d": [0.8279], "cifar100.cifar100_allcnnc": [0.5277], "svhn.svhn_wrn164": [0.8529], "tolstoi.tolstoi_char_rnn": [0.6053]}

    # Dict with convergence performances using test loss
    conv_loss = {"quadratic.noisy_quadratic": [6.41], "mnist.mnist_vae": [76.21], "fmnist.fmnist_vae": [29.76]}

    if args.test_problem in conv_perf:
        return conv_perf[args.test_problem], True
    elif args.test_problem in conv_loss:
        return conv_loss[args.test_problem], False
    else:
        print("Warning: No convergence performance defined for this problem!")
        return 1.0, True
