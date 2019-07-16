"""Example run script using StandardRunner."""

import tensorflow as tf
import deepobs.tensorflow as tfobs

optimizer_class = tf.train.MomentumOptimizer
hyperparams = {"learning_rate": {"type":float},
               "momentum": {"type": float, "default": 0.99},
               "use_nesterov": {"type": bool, "default": False}}

runner = tfobs.runners.StandardRunner(optimizer_class, hyperparams)

# The run method needs the explicit hyperparameter values as a dictionary. All arguments that are not
# provided will automatically be grabbed from the command line or default to their default values.
runner.run(hyperparams={'learning_rate': 0.1}, train_log_interval = 10)
