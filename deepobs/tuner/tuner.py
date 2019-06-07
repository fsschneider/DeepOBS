# -*- coding: utf-8 -*-
import abc
from .. import config

class Tuner(abc.ABC):
    def __init__(self,
                 optimizer_class,
                 hyperparams,
                 ressources,
                 mode = 'final',
                 runner_type = 'StandardRunner'):

        self._optimizer_class = optimizer_class
        self._hyperparams = hyperparams
        self._ressources = ressources
        self._mode = mode

        if config.get_framework == 'tensorflow':
            from .. import tensorflow as fw
        elif config.get_framework == 'pytorch':
            from .. import pytorch as fw
        else:
            raise RuntimeError('Framework not implemented.')
        # check if requested runner is implemented as a class
        try:
            # TODO make sure that tf and pt pathes are consistent
            self._runner = getattr(fw.runners.runner, runner_type)
        except AttributeError:
            raise AttributeError('Runner type ', runner_type,' not implemented. If you really need it, you have to implement it on your own.')

    @abc.abstractmethod
    def tune(self, testproblem, **training_params):
        return

    def tune_small_test_set(self, **training_params):
        testproblems = config.get_small_test_set()
        for testproblem in testproblems:
            self.tune(testproblem, **training_params)

    def tune_large_test_set(self, **training_params):
        testproblems = config.get_large_test_set()
        for testproblem in testproblems:
            self.tune(testproblem, **training_params)