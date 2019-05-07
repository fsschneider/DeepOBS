#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns

import deepobs

sns.set()
sns.set_style("whitegrid", {
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False
})


def plot_table(folder_pars, baseline_pars=None):
    print("Plot overall performance table")

    bm_table_small = dict()
    for testprob in [
            "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
    ]:
        bm_table_small[testprob] = dict()
        bm_table_small[testprob]['Performance'] = dict()
        bm_table_small[testprob]['Speed'] = dict()
        bm_table_small[testprob]['Tuneability'] = dict()
        if testprob in folder_pars.testproblems:
            for _, opt in folder_pars.testproblems[testprob].optimizers.items():
                bm_table_small[testprob] = opt.get_bm_table(
                    bm_table_small[testprob])
        if baseline_pars is not None:
            if testprob in baseline_pars.testproblems:
                for _, opt in baseline_pars.testproblems[
                        testprob].optimizers.items():
                    bm_table_small[testprob] = opt.get_bm_table(
                        bm_table_small[testprob])
    bm_table_small_pd = deepobs.analyzer.analyze_utils.beautify_plot_table(
        bm_table_small)
    deepobs.analyzer.analyze_utils.texify_plot_table(bm_table_small_pd,
                                                     "small")

    bm_table_large = dict()
    for testprob in [
            "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"
    ]:
        bm_table_large[testprob] = dict()
        bm_table_large[testprob]['Performance'] = dict()
        bm_table_large[testprob]['Speed'] = dict()
        bm_table_large[testprob]['Tuneability'] = dict()
        if testprob in folder_pars.testproblems:
            for _, opt in folder_pars.testproblems[testprob].optimizers.items():
                bm_table_large[testprob] = opt.get_bm_table(
                    bm_table_large[testprob])
        if baseline_pars is not None:
            if testprob in baseline_pars.testproblems:
                for _, opt in baseline_pars.testproblems[
                        testprob].optimizers.items():
                    bm_table_large[testprob] = opt.get_bm_table(
                        bm_table_large[testprob])
    bm_table_large_pd = deepobs.analyzer.analyze_utils.beautify_plot_table(
        bm_table_large)
    deepobs.analyzer.analyze_utils.texify_plot_table(bm_table_large_pd,
                                                     "large")
