"""Example run script using StandardRunner."""

import tensorflow as tf

from deepobs import tensorflow as tfobs

optimizer_class = tf.train.MomentumOptimizer
hyperparams = {
    "learning_rate": {"type": float},
    "momentum": {"type": float, "default": 0.99},
    "use_nesterov": {"type": bool, "default": False},
}

runner = tfobs.runners.StandardRunner(optimizer_class, hyperparams)
runner.run()
