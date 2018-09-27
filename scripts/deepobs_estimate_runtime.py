#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs a user-written run script and SGD sequentially and time both to estimate the runtime overhead of the new optimizer
"""
from __future__ import print_function

import argparse
import time
import os
import numpy as np


# ------- Parse Command Line Arguments ----------------------------------------
parser = argparse.ArgumentParser(description="Run a new run script and compare its runtime to SGD.")
parser.add_argument("run_script",
                    help="path to the new run_script.")
parser.add_argument("--test_problem", default='mnist.mnist_mlp',
                    help="Name of the test problem to run both scripts.")
parser.add_argument("--data_dir", default='data_deepobs',
                    help="Path to the base data dir. If not set, deepobs uses its default.")
parser.add_argument("--bs", "--batch_size", default=128, type=int,
                    help="The batch size (positive integer). Defaults to 128.")
parser.add_argument("--lr", "--learning_rate", default=1e-5,
                    help="The learning rate of SGD, defaults to 1e-5.")
# Number of steps and checkpoint interval
parser.add_argument("-N", "--num_epochs", default=3, type=int,
                    help="Total number of training steps.")

parser.add_argument("--num_runs", default=5, type=int,
                    help="Total number of runs for each optimizer.")

parser.add_argument("--saveto", default=None,
                    help="Folder for saving a txt files with a summary.")

parser.add_argument("--optimizer_arguments", help="Additional arguments for the new optimizer", type=str)

args = parser.parse_args()
# -----------------------------------------------------------------------------

SGD_times = []
New_opt_times = []

for i in range(args.num_runs):
    print("** Start Run: ", i)
    # SGD
    run_sgd = "deepobs_run_sgd.py " + args.test_problem + " --bs=" + str(args.bs) + " --lr=" + str(args.lr) + " --num_epochs=" + str(args.num_epochs) + " --data_dir=" + args.data_dir + " --nologs"

    print("Running...", run_sgd)
    start_SGD = time.time()
    os.system(run_sgd)
    end_SGD = time.time()

    SGD_times.append(end_SGD - start_SGD)
    print("Time for SGD run ", i, ": ", SGD_times[-1])

    # New Optimizer
    optimizer_arguments = args.optimizer_arguments.split('--')
    optimizer_arguments_clear = filter(None, optimizer_arguments)

    run_script = "python " + args.run_script + " " + args.test_problem + " --bs=" + str(args.bs) + " --num_epochs=" + str(args.num_epochs) + " --data_dir=" + args.data_dir + " --nologs"

    for arg in optimizer_arguments_clear:
        run_script += " --" + arg

    print("Running...", run_script)
    start_script = time.time()
    os.system(run_script)
    end_script = time.time()

    New_opt_times.append(end_script - start_script)
    print("Time for new optimizer run ", i, ": ", New_opt_times[-1])

overhead = np.divide(New_opt_times, SGD_times)

output = "** Mean run time SGD: " + str(np.mean(SGD_times)) + "\n" + "** Mean run time new optimizer: " + str(np.mean(New_opt_times)) + "\n" + "** Overhead per run: " + str(overhead) + "\n" + "** Mean overhead: " + str(np.mean(overhead)) + " Standard deviation: " + str(np.std(overhead))

print(output)

if args.saveto is not None:
    if not os.path.exists(args.saveto):
        os.makedirs(args.saveto)
    with open(os.path.join(args.saveto, 'estimated_runtime.txt'), 'w') as txt_file:
        txt_file.write(output)
