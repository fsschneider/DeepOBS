"""Simple example of using the analyzer module of DeepOBS."""

from deepobs import analyzer

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
performance_dic = analyzer.get_performance_dictionary("./results/quadratic_deep/SGD")
print(performance_dic)

# plot the training curve for the best performance
analyzer.plot_optimizer_performance("./results/quadratic_deep/SGD")
