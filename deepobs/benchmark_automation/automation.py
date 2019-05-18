# -*- coding: utf-8 -*-

class SmallAutomater(object):
    def __init__(self, framework, optimizer, hyperparams, runner_type = 'StandardRunner'):
        # TODO where to capture the tuned parameters like learning rate?
        self.framework = framework

        if self.framework == 'tensorflow':
            from .. import tensorflow as fw
        elif self.framework == 'pytorch':
            from .. import pytorch as fw
        else:
            raise RuntimeError('framework not implemented')

        # check if requested runner is implemented as a class
        try:
            # TODO make sure that tf and pt pathes are consistent
            # TODO how to refer to the same file every time?
            self.runner = getattr(fw.runners.standard_runner, runner_type)
        except AttributeError:
            raise AttributeError('Runner type not implemented. If you really need it, you have to implement it on your own.')

        self.optimizer = optimizer
        self.hyperparams = hyperparams
        # init the runner
        self.runner = self.runner(optimizer, hyperparams)

        self.testproblems = {'quadratic_deep':  {'batch_size': 128,
                                                'num_epochs:': 100},
                             'mnist_vae':       {'batch_size': 64,
                                                 'num_epochs:': 50},
                             'fmnist_2c2d':     {'batch_size': 128,
                                                 'num_epochs:': 100},
                             'cifar10_3c3d':    {'batch_size': 128,
                                                 'num_epochs:': 100}
                             }

        self.general_settings = {'train_log_interval': 10,
                                 'learning_rate': 0.1
                                 }

    def run(self):
        for testproblem, testproblem_settings in self.testproblems.items():
            # combine general settings and testproblem specific settings
            settings = {**self.general_settings, **testproblem_settings}
            self.runner.run(testproblem=testproblem, **settings)