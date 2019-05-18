"""Example analyze script using the Analyzer Class."""

import deepobs

# create the analyzer object
ana = deepobs.analyzer.Analyzer('./results',
                                metric='test_accuracies')

# print the best settings for all modes
ana.print_best_runs()

# plot a basic performance curve
fig, axes = ana.plot_performance(mode='final')
