#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs a user-written run script and SGD sequentially and time both to estimate
the runtime overhead of the new optimizer
"""
from __future__ import print_function

import argparse
import time
import os
import numpy as np
import tensorflow as tf

import deepobs.tensorflow as tfobs


# ------- Parse Command Line Arguments ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a new run script and compare its runtime to SGD.")
    parser.add_argument("run_script", help="Path to the new run_script.")
    parser.add_argument(
        "--test_problem",
        default='mnist_mlp',
        help="Name of the test problem to run both scripts.")
    parser.add_argument(
        "--data_dir",
        default='data_deepobs',
        help="Path to the base data dir. If not set, deepobs uses its default."
    )
    parser.add_argument(
        "--bs",
        "--batch_size",
        default=128,
        type=int,
        help="The batch size (positive integer).")
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=1e-5,
        help=
        "The learning rate of both SGD and the new optimizer, defaults to 1e-5."
    )
    # Number of steps and checkpoint interval
    parser.add_argument(
        "-N",
        "--num_epochs",
        default=3,
        type=int,
        help="Total number of training epochs per run.")

    parser.add_argument(
        "--num_runs",
        default=5,
        type=int,
        help="Total number of runs for each optimizer.")

    parser.add_argument(
        "--saveto",
        default=None,
        help="Folder for saving a txt files with a summary.")

    parser.add_argument(
        "--optimizer_args",
        help="Additional arguments for the new optimizer",
        type=str)

    return parser


def read_args():
    parser = parse_args()
    args = parser.parse_args()
    return args


def main(run_script,
         optimizer_args,
         test_problem="mnist.mnist_mlp",
         data_dir="data_deepobs",
         bs=128,
         lr=1e-5,
         num_epochs=3,
         num_runs=5,
         saveto=None):
    # Put all input arguments back into an args variable, so I can use it as before (without the main function)
    args = argparse.Namespace(**locals())
    SGD_times = []
    New_opt_times = []

    for i in range(args.num_runs):
        print("** Start Run: ", i + 1, "of", args.num_runs)

        # SGD

        print("Running SGD")
        start_SGD = time.time()
        runner = tfobs.runners.StandardRunner(
            tf.train.GradientDescentOptimizer, [])
        runner._run(
            testproblem=args.test_problem,
            weight_decay=None,
            batch_size=args.bs,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            lr_sched_epochs=None,
            lr_sched_factors=None,
            random_seed=42,
            data_dir=args.data_dir,
            output_dir=None,
            train_log_interval=10,
            print_train_iter=False,
            tf_logging=False,
            no_logs=True)
        end_SGD = time.time()

        SGD_times.append(end_SGD - start_SGD)
        print("Time for SGD run ", i + 1, ": ", SGD_times[-1])

        # New Optimizer
        run_script = "python " + args.run_script + " " + args.test_problem + " --lr=" + str(
            args.lr) + " --bs=" + str(args.bs) + " --num_epochs=" + str(
                args.
                num_epochs) + " --data_dir=" + args.data_dir + " --no_logs"
        # add optimizer_args if necessary
        if args.optimizer_args is not None:
            optimizer_args = args.optimizer_args.split('--')
            optimizer_args_clear = filter(None, optimizer_args)
            for arg in optimizer_args_clear:
                run_script += " --" + arg

        print("Running...", run_script)
        start_script = time.time()
        os.system(run_script)
        end_script = time.time()

        New_opt_times.append(end_script - start_script)
        print("Time for new optimizer run ", i + 1, ": ", New_opt_times[-1])

    overhead = np.divide(New_opt_times, SGD_times)

    output = "** Mean run time SGD: " + str(
        np.mean(SGD_times)) + "\n" + "** Mean run time new optimizer: " + str(
            np.mean(New_opt_times)) + "\n" + "** Overhead per run: " + str(
                overhead) + "\n" + "** Mean overhead: " + str(
                    np.mean(overhead)) + " Standard deviation: " + str(
                        np.std(overhead))

    print(output)


if __name__ == '__main__':
    main(**vars(read_args()))
