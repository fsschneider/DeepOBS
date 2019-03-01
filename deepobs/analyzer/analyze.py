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


def get_best_run(folder_pars):
    print("Get best run\n\n")
    for _, testprob in folder_pars.testproblems.items():
        print("***********************")
        print("Analyzing", testprob.name)
        print("***********************")
        for _, opt in testprob.optimizers.items():
            #         print("Analyzing", opt.name)
            print("Checked", opt.num_settings, "settings for", opt.name,
                  "and found the following")
            setting_final = opt.get_best_setting_final()
            setting_best = opt.get_best_setting_best()
            print("Best Setting (Final Value)", setting_final.name,
                  "with final performance of",
                  setting_final.aggregate.final_value)
            print("Best Setting (Best Value)", setting_best.name,
                  "with best performance of",
                  setting_best.aggregate.best_value)


def plot_lr_sensitivity(folder_pars, baseline_pars=None, mode='final'):
    print("Plot learning rate sensitivity plot")
    fig, axis = plt.subplots(2, 4, figsize=(35, 4))

    ax_row = 0
    for testprob in [
            "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
    ]:
        if testprob in folder_pars.testproblems:
            for _, opt in folder_pars.testproblems[testprob].optimizers.items(
            ):
                opt.plot_lr_sensitivity(axis[0][ax_row], mode=mode)
            ax_row += 1
    if baseline_pars is not None:
        ax_row = 0
        for testprob in [
                "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
        ]:
            if testprob in baseline_pars.testproblems:
                for _, opt in baseline_pars.testproblems[
                        testprob].optimizers.items():
                    opt.plot_lr_sensitivity(axis[0][ax_row], mode=mode)
                ax_row += 1
    ax_row = 0
    for testprob in [
            "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"
    ]:
        if testprob in folder_pars.testproblems:
            for _, opt in folder_pars.testproblems[testprob].optimizers.items(
            ):
                opt.plot_lr_sensitivity(axis[1][ax_row], mode=mode)
            ax_row += 1
    if baseline_pars is not None:
        ax_row = 0
        for testprob in [
                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164",
                "tolstoi_char_rnn"
        ]:
            if testprob in baseline_pars.testproblems:
                for _, opt in baseline_pars.testproblems[
                        testprob].optimizers.items():
                    opt.plot_lr_sensitivity(axis[1][ax_row], mode=mode)
                ax_row += 1

    fig, axis = deepobs.analyzer.analyze_utils.beautify_lr_sensitivity(
        fig, axis)
    deepobs.analyzer.analyze_utils.texify_lr_sensitivity(fig, axis)
    plt.show()


def plot_performance(folder_pars, baseline_pars=None, mode="most"):
    # Small Benchmark
    fig, axis = plt.subplots(4, 4, sharex='col', figsize=(25, 8))

    ax_col = 0
    for testprob in [
            "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
    ]:
        if testprob in folder_pars.testproblems:
            for _, opt in folder_pars.testproblems[testprob].optimizers.items(
            ):
                opt.plot_performance(axis[:, ax_col], mode=mode)
            ax_col += 1
    if baseline_pars is not None:
        ax_col = 0
        for testprob in [
                "quadratic_deep", "mnist_vae", "fmnist_2c2d", "cifar10_3c3d"
        ]:
            if testprob in baseline_pars.testproblems:
                for _, opt in baseline_pars.testproblems[
                        testprob].optimizers.items():
                    opt.plot_performance(axis[:, ax_col], mode='most')
                ax_col += 1
    fig, axis = deepobs.analyzer.analyze_utils.beautify_plot_performance(
        fig, axis, folder_pars, "small")
    deepobs.analyzer.analyze_utils.texify_plot_performance(fig, axis, "small")
    plt.show()

    # Large Benchmark
    fig, axis = plt.subplots(4, 4, sharex='col', figsize=(25, 8))

    ax_col = 0
    for testprob in [
            "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164", "tolstoi_char_rnn"
    ]:
        if testprob in folder_pars.testproblems:
            for _, opt in folder_pars.testproblems[testprob].optimizers.items():
                opt.plot_performance(axis[:, ax_col], mode=mode)
            ax_col += 1
    if baseline_pars is not None:
        ax_col = 0
        for testprob in [
                "fmnist_vae", "cifar100_allcnnc", "svhn_wrn164",
                "tolstoi_char_rnn"
        ]:
            if testprob in baseline_pars.testproblems:
                for _, opt in baseline_pars.testproblems[
                        testprob].optimizers.items():
                    opt.plot_performance(axis[:, ax_col], mode='most')
                ax_col += 1
    fig, axis = deepobs.analyzer.analyze_utils.beautify_plot_performance(
        fig, axis, folder_pars, "large")
    deepobs.analyzer.analyze_utils.texify_plot_performance(fig, axis, "large")
    plt.show()


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
