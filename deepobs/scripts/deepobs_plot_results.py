#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import deepobs


def parse_args():
    parser = argparse.ArgumentParser(description="Plotting tool for DeepOBS.")
    parser.add_argument("path", help="Path to the results folder")
    parser.add_argument(
        "--get_best_run",
        action="store_const",
        const=True,
        default=False,
        help="Return best hyperparameter setting per optimizer and testproblem."
    )
    parser.add_argument(
        "--plot_lr_sensitivity",
        action="store_const",
        const=True,
        default=False,
        help="Plot 'sensitivity' plot for the learning rates.")
    parser.add_argument(
        "--plot_performance",
        action="store_const",
        const=True,
        default=False,
        help="Plot performance plot compared to the baselines.")
    parser.add_argument(
        "--plot_table",
        action="store_const",
        const=True,
        default=False,
        help=
        "Plot overall performance table including speed and hyperparameters.")
    parser.add_argument(
        "--full",
        action="store_const",
        const=True,
        default=False,
        help="Run a full analysis and plot all figures.")
    parser.add_argument(
        "--baseline_path",
        type=str,
        default="baselines_deepobs",
        help="Path to baseline folder.")
    return parser


def read_args():
    parser = parse_args()
    args = parser.parse_args()
    return args


def main(path, get_best_run, plot_lr_sensitivity, plot_performance, plot_table,
         full, baseline_path):
    # Put all input arguments back into an args variable, so I can use it as
    # before (without the main function)
    args = argparse.Namespace(**locals())
    # Parse whole baseline folder
    if args.baseline_path:
        print("Parsing baseline folder")
        deepobs.tensorflow.config.set_baseline_dir(args.baseline_path)
        baseline_parser = deepobs.analyzer.analyze_utils.Analyzer(
            deepobs.tensorflow.config.get_baseline_dir())
    else:
        baseline_parser = None

    # Parse path folder
    print("Parsing results folder")
    folder_parser = deepobs.analyzer.analyze_utils.Analyzer(args.path)

    if args.get_best_run or args.full:
        deepobs.analyzer.analyze.get_best_run(folder_parser)
    if args.plot_lr_sensitivity or args.full:
        deepobs.analyzer.analyze.plot_lr_sensitivity(folder_parser,
                                                     baseline_parser)
    if args.plot_performance or args.full:
        deepobs.analyzer.analyze.plot_performance(folder_parser,
                                                  baseline_parser)
    if args.plot_table or args.full:
        deepobs.analyzer.analyze.plot_table(folder_parser, baseline_parser)


if __name__ == '__main__':
    main(**vars(read_args()))
