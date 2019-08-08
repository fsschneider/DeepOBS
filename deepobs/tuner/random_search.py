# -*- coding: utf-8 -*-
from .tuner import ParallelizedTuner


class RandomSearch(ParallelizedTuner):
    """A basic Random Search tuner.
    """
    def __init__(self, optimizer_class, hyperparam_names, distributions, ressources, runner):
        """
        Args:
            distributions (dict): Holds the distributions for each hyperparameter.\
            Each distribution must implement an rvs() method to draw random variates.\
            For instance, all scipy.stats.distribution distributions are applicable.
        """
        super(RandomSearch, self).__init__(optimizer_class, hyperparam_names, ressources, runner)

        self._distributions = distributions
        self._search_name = 'random_search'

    def _sample(self):
        params = []
        for i in range(self._ressources):
            # sample parameters
            sample = {}
            for param_name, param_distr in self._distributions.items():
                sample[param_name] = param_distr.rvs()
            params.append(sample)
        return params
