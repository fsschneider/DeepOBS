=============================
Learning Rate Schedule Runner
=============================

Using the learning rate schedule runner adds two more training parameters to the training loop, the epochs and factors for the learning rate decay. The example below shows how to use it in a run file, but these parameters are also automatically added to be command line arguments.

.. code-block:: python

  optimizer_class = tf.train.MomentumOptimizer
  hyperparms = {'lr': {'type': float},
	'momentum': {'type': float, 'default': 0.99},
	'uses_nesterov': {'type': bool, 'default': False}}
  schedule = {
            "name": "step",
            "lr_sched_epochs": [2, 4],
            "lr_sched_factors": [0.1, 0.01]
        }
  runner = tfobs.runners.LearningRateScheduleRunner(optimizer_class, hyperparams)
  runner.run(testproblem='quadratic_deep', hyperparams={'learning_rate': 1e-2}, num_epochs=10, lr_sched_epochs=schedule["lr_sched_epochs"], lr_sched_factors=schedule["lr_sched_factors"])


.. currentmodule:: deepobs.tensorflow.runners

.. autoclass:: LearningRateScheduleRunner
    :members:
    :inherited-members:
    :show-inheritance:
    :special-members: __init__
