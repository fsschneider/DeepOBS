# -*- coding: utf-8 -*-
from .tuner import Tuner
from bayes_opt import UtilityFunction
import bayes_opt
from .bayesian_utils import _save_bo_optimizer_object, _init_summary_directory, _update_bo_tuning_summary
import os
from .tuner_utils import rerun_setting


class GP(Tuner):
    """A Bayesian optimization tuner that uses a Gaussian Process surrogate.
    """
    def __init__(self, optimizer_class,
                 hyperparam_names,
                 bounds,
                 ressources,
                 runner,
                 transformations = None):
        """
        Args:
            optimizer_class (framework optimizer class): The optimizer to tune.
            hyperparam_names (dict): Nested dictionary that holds the name, type and default values of the hyperparameters
            bounds (dict): A dict where the key is the hyperparameter name and the value is a tuple of its bounds.
            ressources (int): The number of total evaluations of the tuning process.
            transformations (dict): A dict where the key is the hyperparameter name and the value is a callable that returns \
            the transformed hyperparameter.
            runner: The DeepOBS runner which is used for each evaluation.
        """
        super(GP, self).__init__(optimizer_class, hyperparam_names, ressources, runner)
        self._bounds = bounds
        self._transformations = transformations

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

    def _generate_cost_function(self, testproblem, output_dir, mode, **kwargs):
        def _cost_function(**hyperparams):
            # apply transformations if they exist
            if self._transformations is not None:
                for hp_name, hp_transform in self._transformations.items():
                    hyperparams[hp_name] = hp_transform(hyperparams[hp_name])
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
                
    def tune(self, testproblem,
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
             rerun_best_setting = False,
             **kwargs):

        """Tunes the optimizer hyperparameters by evaluating a Gaussian process surrogate with an acquisition function.
        Args:
            testproblem (str): The test problem to tune the optimizer on.
            output_dir (str): The output directory for the results.
            random_seed (int): Random seed for the whole truning process. Every individual run is seeded by it.
            n_init_samples (int): The number of random exploration samples in the beginning of the tuning process.
            tuning_summary (bool): Whether to write an additional tuning summary. Can be used to get an overview over the tuning progress
            plotting_summary (bool): Whether to store additional objects that can be used to plot the posterior.
            kernel (Sklearn.gaussian_process.kernels.Kernel): The kernel of the GP.
            acq_type (str): The type of acquisition function to use. Muste be one of ``ucb``, ``ei``, ``poi``.
            acq_kappa (float): Scaling parameter of the acquisition function.
            acq_xi (float): Scaling parameter of the acquisition function.
            mode (str): The mode that is used to evaluate the cost. Must be one of ``final`` or ``best``.
            rerun_best_setting (bool): Whether to automatically rerun the best setting with 10 different seeds.
            """

        self._set_seed(random_seed)
        log_path = os.path.join(output_dir, testproblem, self._optimizer_name)

        cost_function = self._generate_cost_function(testproblem, output_dir, mode, **kwargs)

        op = bayes_opt.BayesianOptimization(f = None, pbounds = self._bounds, random_state=random_seed)
        if kernel is not None:
            op._gp.kernel = kernel

        utility_func = UtilityFunction(acq_type, kappa = acq_kappa, xi = acq_xi)

        if tuning_summary:
            _init_summary_directory(log_path, 'bo_tuning_log.json')
        if plotting_summary:
            _init_summary_directory(os.path.join(log_path, 'obj'))
            _save_bo_optimizer_object(os.path.join(log_path, 'obj'), 'acq_func', utility_func)

        # evaluates the random points
        try:
            assert n_init_samples <= self._ressources
        except AssertionError:
            raise AssertionError('Number of initial evaluations exceeds the ressources.')
        self._init_bo_space(op,
                            cost_function,
                            n_init_samples,
                            log_path,
                            plotting_summary,
                            tuning_summary)

        # execute remaining ressources
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

        if rerun_best_setting:
            optimizer_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            rerun_setting(self._runner, self._optimizer_class, self._hyperparam_names, optimizer_path)
