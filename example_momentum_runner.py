"""Example run script using StandardRunner."""

import tensorflow as tf
import deepobs.tensorflow as tfobs

optimizer_class = tf.train.MomentumOptimizer
hyperparams = [{
    "name": "momentum",
    "type": float
}, {
    "name": "use_nesterov",
    "type": bool,
    "default": False
}]
runner = tfobs.runners.StandardRunner(optimizer_class, hyperparams)

# The run method accepts all the relevant inputs, all arguments that are not
# provided will automatically be grabbed from the command line.
runner.run(train_log_interval=10)
