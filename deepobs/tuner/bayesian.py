# -*- coding: utf-8 -*-
from .tuner import Tuner
from bayes_opt import UtilityFunction
import bayes_opt
from .bayesian_utils import _init_bo_plotting_summary, _init_bo_tuning_summary, _update_bo_plotting_summary, _update_bo_tuning_summary
import os

class GP(Tuner):
    def __init__(self, optimizer_class,
                 bounds,
                 ressources,
                 acquisition_function = 'EI',
                 runner_type='StandardRunner'):

    # TODO ressources and acq should rather be a argument of tune()\
    # TODO if tune() is the only thing a tuner does, then why have a class?
    
        hyperparams = sorted(bounds)
        super(GP, self).__init__(optimizer_class, hyperparams, ressources, runner_type)

        self._acquisition_function = acquisition_function
        self._bounds = self._read_in_bounds(bounds)

    @staticmethod
    def _determine_cost_from_output_and_mode(output, mode):
        # check if accuracy is available, else take loss
        if not 'test_accuracies' in output:
            # -1 because bayes_opt looks for maximum of cost function
            cost = [-1*v for v in output['test_losses']]
        else:
            cost = output['test_accuracies']

        if mode == 'final':
            cost = cost[-1]
        elif mode == 'best':
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
    # TODO how to deal with discrete categoricals?
    def _generate_cost_function(self, testproblem, output_dir, mode, **kwargs):
        '''Factory to create the cost function depending on the testproblem and kwargs.'''
        def _cost_function(**hyperparams):
            runner = self._runner(self._optimizer_class)
            output = runner.run(testproblem, hyperparams, output_dir=output_dir, **kwargs)
            cost = self._determine_cost_from_output_and_mode(output, mode)
            return cost
        return _cost_function
    
    def _init_bo_space(self, op, cost_function, n_init_samples):
        for iteration in range(1, n_init_samples+1):
            random_sample = op.space.random_sample()
            params = dict(zip(self._hyperparams, random_sample))
            target = cost_function(**params)
            op.register(params, target)
        # finally fit the gp
        op._gp.fit(op._space.params, op._space.target)
        
    def tune(self, testproblems, 
             output_dir = './results', 
             random_seed = 42, 
             n_init_samples = 5, 
             tuning_summary = True, 
             plotting_summary = True, 
             kernel = None, 
             acq_type = 'ucb', 
             acq_kappa = 2.576, 
             acq_xi = 0.0, 
             mode = 'final',
             **kwargs):
        
        self._set_seed(random_seed)
        # TODO noise level of the cost function?
        testproblems = self._read_testproblems(testproblems)
        for testproblem in testproblems:
            cost_function = self._generate_cost_function(testproblem, output_dir, mode, **kwargs)
            # TODO when to normalize the y values in gp ?
            op = bayes_opt.BayesianOptimization(f = None, pbounds = self._bounds, random_state=random_seed)
            if kernel is not None:
                # TODO how to check if kernel is valid
                op._gp.kernel = kernel
            
            log_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            self._check_output_path(log_path)
            
            # evaluates the random points
            self._init_bo_space(op, cost_function, n_init_samples)
            utility_func = UtilityFunction(acq_type, kappa = acq_kappa, xi = acq_xi)
            
            if plotting_summary:
                domain = _init_bo_plotting_summary(utility_func, op._gp, self._bounds, log_path)
                
            if tuning_summary:
                _init_bo_tuning_summary(log_path, op)
            
            for iteration in range(1, self._ressources+1):
                next_point = op.suggest(utility_func)
                actual_target = cost_function(**next_point)
                op.register(params=next_point, target = actual_target)
                
                if tuning_summary:
                    _update_bo_tuning_summary(op, iteration, log_path)
                
                # fit gp on new registered points
                op._gp.fit(op._space.params, op._space.target)
                if plotting_summary:
                    _update_bo_plotting_summary(utility_func, op._gp, iteration, domain, log_path)
        return op