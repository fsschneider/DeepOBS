# -*- coding: utf-8 -*-
from .tuner import ParallelizedTuner
class RandomSearch(ParallelizedTuner):

    def __init__(self, optimizer_class, hyperparams, ressources, distributions, runner_type='StandardRunner'):
        super(RandomSearch, self).__init__(optimizer_class, hyperparams, ressources, runner_type)

        self._distributions = distributions

    def _sample(self):
        params = []
        for i in range(self._ressources):
            # sample parameters
            sample = {}
            for param_name, param_distr in self._distributions.items():
                sample[param_name] = param_distr.rvs()
            params.append(sample)
        return params