from deepobs import analyzer

# get the overall best performance of the MomentumOptimizer on the quadratic_deep testproblem
performance_dic = analyzer.get_performance_dictionary('./results/quadratic_deep/SGD')
print(performance_dic)

# plot the training curve for the best performance
analyzer.plot_optimizer_performance('./results/quadratic_deep/SGD')

# plot again, but this time compare to the Adam baseline
analyzer.plot_optimizer_performance('./results/quadratic_deep/SGD',
                                    reference_path='../DeepOBS_Baselines/baselines_tensorflow/quadratic_deep/MomentumOptimizer')
