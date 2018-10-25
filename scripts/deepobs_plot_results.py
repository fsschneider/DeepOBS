#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib2tikz import save as tikz_save
import matplotlib as mpl
import collections
import pandas as pd
import argparse
import deepobs


# ------- Parse Command Line Arguments ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Plot results using deepobs")
    parser.add_argument("--results_dir",
                        help="Path to the base result dir.")
    parser.add_argument("--saveto", default=None,
                        help="Folder for saving the resulting png and tex files.")
    parser.add_argument("--log", action="store_const",
                        const=True, default=False,
                        help="Add this flag to plot using a logarithmic y axis.")
    return parser


def read_args():
    parser = parse_args()
    args = parser.parse_args()
    return args
# -----------------------------------------------------------------------------


def main(results_dir, saveto=None, log=False):
    # Put all input arguments back into an args variable, so I can use it as before (without the main function)
    args = argparse.Namespace(**locals())
    # Set Seaborn for nicer plots
    sns.set()
    sns.set_context('paper')
    # Look:
    lw = mpl.rcParams['lines.linewidth']
    lc = ['#A51E37', '#B4A069', '#0069AA', '#7DA54B', '#AF6E96']
    ls = ["--", "-.", ":"]  # linestyle for baselines
    baseline_color = '#AFB3B7'
    cm = sns.light_palette("green", as_cmap=True)

    # Create empty performance table (keeping score of performance, speed, tuneability)
    perf_table = collections.OrderedDict()

    file_structure = deepobs.plot_utils.get_filestructure(args.results_dir)

    # Loop over test problems
    for prob, opt in file_structure.iteritems():
        test_problem = deepobs.plot_utils.get_test_problem(opt.itervalues().next()[0])
        print("Working on", test_problem)
        baselines = deepobs.plot_utils.get_baselines(test_problem)
        if not baselines:
            print("Sorry, I have no baselines for ", test_problem, " at the moment.")
        f, ax = deepobs.plot_utils.create_figure(name=test_problem)
        # Create performance Table for this test problem
        perf_table[deepobs.plot_utils.texify(test_problem)] = collections.OrderedDict()
        perf_table[deepobs.plot_utils.texify(test_problem)]['Performance'] = collections.OrderedDict()
        perf_table[deepobs.plot_utils.texify(test_problem)]['Speed'] = collections.OrderedDict()
        perf_table[deepobs.plot_utils.texify(test_problem)]['Tuneability'] = collections.OrderedDict()

        # Loop over baselines
        for bsl_idx, baseline_optimizer in enumerate(baselines):
            baselines[baseline_optimizer].plot(ax, lc=baseline_color, lw=0.5 * lw, ls=ls[bsl_idx])
            # Add to performance table
            perf_table[deepobs.plot_utils.texify(baselines[baseline_optimizer].test_problem)]['Performance'][baselines[baseline_optimizer].name] = baselines[baseline_optimizer].final_performance
            perf_table[deepobs.plot_utils.texify(baselines[baseline_optimizer].test_problem)]['Speed'][baselines[baseline_optimizer].name] = baselines[baseline_optimizer].speed
            perf_table[deepobs.plot_utils.texify(baselines[baseline_optimizer].test_problem)]['Tuneability'][baselines[baseline_optimizer].name] = baselines[baseline_optimizer].opt_args

        # Loop over "new" optimizers
        opt_idx = 0
        for opt, runs in opt.iteritems():
            # optimizer_run is type optimizer run with std dev computed from all runs
            name = str(opt)
            if name.startswith('.'):
                pass
            else:
                avg_optimizer_run = deepobs.plot_utils.get_average_run(runs, name)

                avg_optimizer_run.plot(ax, lc=lc[opt_idx], lw=0.85 * lw, ls='-')
                # Add to performance table
                perf_table[deepobs.plot_utils.texify(avg_optimizer_run.test_problem)]['Performance'][avg_optimizer_run.name] = avg_optimizer_run.final_performance
                perf_table[deepobs.plot_utils.texify(avg_optimizer_run.test_problem)]['Speed'][avg_optimizer_run.name] = avg_optimizer_run.speed
                perf_table[deepobs.plot_utils.texify(avg_optimizer_run.test_problem)]['Tuneability'][avg_optimizer_run.name] = avg_optimizer_run.opt_args
                opt_idx += 1

        # Make clean output
        deepobs.plot_utils.set_figure(ax, args.log)

        if args.saveto is not None:
            if not os.path.exists(args.saveto):
                os.makedirs(args.saveto)
            plt.savefig(os.path.join(args.saveto, test_problem + '.png'))
            tikz_save(os.path.join(args.saveto, test_problem + '.tex'), figureheight='\\figureheight', figurewidth='\\figurewidth')
        else:
            plt.show()

    # Create pandas dataframe from performance table dict
    perf_table_pd = pd.DataFrame.from_dict({(i, j): perf_table[i][j] for i in perf_table.keys() for j in perf_table[i].keys()}, orient='index')
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    if args.saveto is not None:
        if not os.path.exists(args.saveto):
            os.makedirs(args.saveto)
        # Postprocessing for Latex Output
        perf_table_pd_n = perf_table_pd.apply(deepobs.plot_utils.norm, axis=1)  # normalize between 0 and 100
        perf_table_pd_n_str = perf_table_pd_n.applymap(deepobs.plot_utils.add_color_coding_tex) + perf_table_pd.applymap(deepobs.plot_utils.latex)  # combine normalise version with latex color code command
        perf_table_pd_n_str.columns = perf_table_pd_n_str.columns.str.replace('_', '')  # Texify the column headers
        with open(os.path.join(args.saveto, 'performance_table.tex'), 'w') as tex_file:
            tex_file.write("\def\cca#1#2{\cellcolor{green!#1!red}\ifnum #1<50\color{white}\\fi{#2}}" + perf_table_pd_n_str.to_latex(escape=False))
    else:
        print(perf_table_pd.to_string())


if __name__ == '__main__':
    main(**vars(read_args()))


if __name__ == '__main__':
    main(**vars(read_args()))
