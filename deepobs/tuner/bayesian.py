# -*- coding: utf-8 -*-
from .tuner import Tuner
from bayes_opt import UtilityFunction
import bayes_opt
from .bayesian_utils import _save_bo_optimizer_object, _init_bo_tuning_summary, _update_bo_tuning_summary
import os


class GP(Tuner):
    def __init__(self, optimizer_class,
                 hyperparam_names,
                 bounds,
                 ressources,
                 acquisition_function = 'EI',
                 runner_type='StandardRunner'):

    # TODO ressources and acq should rather be a argument of tune()\
    # TODO if tune() is the only thing a tuner does, then why have a class?

        super(GP, self).__init__(optimizer_class, hyperparam_names, ressources, runner_type)

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
        # TODO categoricals?
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
        # TODO das Abbrechen eines Runners wegen NaNs hat Einfluss auf BO results!!!!
        def _cost_function(**hyperparams):
            runner = self._runner(self._optimizer_class, self._hyperparam_names)
            output = runner.run(testproblem, hyperparams, output_dir=output_dir, **kwargs)
            cost = self._determine_cost_from_output_and_mode(output, mode)
            return cost
        return _cost_function
    
    def _init_bo_space(self, 
                       op, 
                       cost_function, 
                       n_init_samples,
                       log_path, 
                       plotting_summary, 
                       tuning_summary):
        
        for iteration in range(1, n_init_samples+1):
            random_sample = op.space.random_sample()
            params = dict(zip(sorted(self._hyperparam_names), random_sample))
            target = cost_function(**params)
            if tuning_summary:
                _update_bo_tuning_summary(op._gp, params, target, log_path)
            op.register(params, target)
            
            # fit gp on new registered points
            op._gp.fit(op._space.params, op._space.target)
            if plotting_summary:
                _save_bo_optimizer_object(os.path.join(log_path, 'obj'), str(iteration), op)
                
    def tune(self, testproblems, 
             output_dir = './results', 
             random_seed = 42, 
             n_init_samples = 5, 
             tuning_summary = True, 
             plotting_summary = True, 
             kernel = None,
             alpha = None,
             acq_type = 'ucb', 
             acq_kappa = 2.576, 
             acq_xi = 0.0, 
             mode = 'final',
             **kwargs):
        
        testproblems = self._read_testproblems(testproblems)
        for testproblem in testproblems:
            self._set_seed(random_seed)
            cost_function = self._generate_cost_function(testproblem, output_dir, mode, **kwargs)
            # TODO when to normalize the y values in gp ?
            op = bayes_opt.BayesianOptimization(f = None, pbounds = self._bounds, random_state=random_seed)
            if kernel is not None:
                # TODO how to check if kernel is valid
                op._gp.kernel = kernel
            if alpha is not None:
                # set noise level
                op._gp.alpha = alpha
            
            log_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            self._check_output_path(log_path)
            
            utility_func = UtilityFunction(acq_type, kappa = acq_kappa, xi = acq_xi)

            if tuning_summary:
                _init_bo_tuning_summary(log_path, op)            
            if plotting_summary:
                self._check_output_path(os.path.join(log_path, 'obj'))
                _save_bo_optimizer_object(os.path.join(log_path, 'obj'), 'acq_func', utility_func)
                
            # evaluates the random points
            self._init_bo_space(op, 
                                cost_function, 
                                n_init_samples,
                                log_path, 
                                plotting_summary, 
                                tuning_summary)
            
            # execute remainig ressources
            # TODO assert that ressources >= n init points
            for iteration in range(n_init_samples+1, self._ressources + 1):
                next_point = op.suggest(utility_func)
                target = cost_function(**next_point)
                if tuning_summary:
                    _update_bo_tuning_summary(op._gp, next_point, target, log_path)
                op.register(params=next_point, target = target)
                
                # fit gp on new registered points
                op._gp.fit(op._space.params, op._space.target)
                if plotting_summary:
                    _save_bo_optimizer_object(os.path.join(log_path, 'obj'), str(iteration), op)
        return op
