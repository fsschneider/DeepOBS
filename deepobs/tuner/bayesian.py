# -*- coding: utf-8 -*-
from .tuner import Tuner
from bayes_opt import UtilityFunction
import bayes_opt
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from .tuner_utils import AdvancedObserver

class GP(Tuner):
    def __init__(self, optimizer_class,
                 bounds,
                 ressources,
                 mode = 'final',
                 acquisition_function = 'EI',
                 runner_type='StandardRunner'):

# TODO if the same tuning run is done again but with e.g. different acquisition function the results folder is already filled with previous runs!
        # make sure to use another folder in this case
        hyperparams = bounds.keys()
        super(GP, self).__init__(optimizer_class, hyperparams, ressources, runner_type)

        self._acquisition_function = acquisition_function
        self._bounds = self._read_in_bounds(bounds)
        self._mode = mode

    def _determine_cost_from_output_and_mode(self, output):
        # check if accuracy is available, else take loss
        # check must work for both frameworks
        if all(v==0 for v in output['test_accuracies']) or not output['test_accuracies']:
            # -1 because bayes_opt looks for maximum of cost function
            cost = [-1*v for v in output['test_losses']]
        else:
            cost = output['test_accuracies']

        if self._mode == 'final':
            cost = cost[-1]
        elif self._mode == 'best':
            cost = max(cost)
        else:
            raise NotImplementedError('''Mode not available for this tuning method. Please use final or best.''')
        return cost

    @staticmethod
    def _read_in_bounds(bounds):
#        bounds = _check_for_categorical(bounds)
        return bounds

    @staticmethod
    def _read_categorical(bounds):
        for key, value in bounds.items():
            if len(value) == 2 and type(value[0]) == bool and type(value[1]) == bool:
                bounds[key] = (0,1)
        return bounds
    # TODO abstract this
    # TODO how to deal with discrete categoricals?
    def _generate_cost_function(self, testproblem, **kwargs):
        '''Factory to create the cost function depending on the testproblem and kwargs.'''
        def _cost_function(**hyperparams):
#            hyperparams = self._build_hyperparams_dict_from_proposal2(proposal_array)
            runner = self._runner(self._optimizer_class)
            output = runner.run(testproblem, hyperparams, **kwargs)
            cost = self._determine_cost_from_output_and_mode(output)
            return cost
        return _cost_function

    def tune(self, testproblems, random_seed = 42, posterior_domain = None, **kwargs):
        self._set_seed(random_seed)
        # testproblems can also be only one testproblem
        if type(testproblems) == str:
            testproblems=testproblems.split()

        for testproblem in testproblems:
            cost_function = self._generate_cost_function(testproblem, **kwargs)
            op = bayes_opt.BayesianOptimization(f = cost_function, pbounds = self._bounds, random_state=random_seed)

            logger = AdvancedObserver(path="./logs.json", bounds = self._bounds, posterior_domain=posterior_domain)
            op.subscribe(bayes_opt.event.Events.OPTMIZATION_STEP, logger)

            op.maximize(init_points=3, n_iter=self._ressources)
        return