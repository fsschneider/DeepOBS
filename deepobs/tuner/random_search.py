# -*- coding: utf-8 -*-
from .tuner import Tuner
from .. import config
class RandomSearch(Tuner):

    def __init__(self, optimizer_class, hyperparams, ressources, distributions, mode = 'final', testproblems=None, runner_type='StandardRunner'):
        super(RandomSearch).__init__(self, optimizer_class, hyperparams, ressources, mode, testproblems, runner_type)

        self._distributions = distributions
    def tune(self, testproblem, **training_params):
        # TODO parallelize
        # TODO return the commands if no parallelization possible
        for i in range(self._ressources):
            # sample parameters
            params = {}
            for param_name, param_distr in self._distributions.iteritems():
                params[param_name] = param_distr.rvs()

            runner = self._runner(self._optimizer_class, params)
            result = runner.run(testproblem, **config.get_testproblem_default_setting(testproblem), **training_params)
        return

