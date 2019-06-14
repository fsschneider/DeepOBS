# -*- coding: utf-8 -*-
from .tuner import Tuner
from GPyOpt.methods import BayesianOptimization as BO

class GP(Tuner):
    def __init__(self, optimizer_class,
                 bounds,
                 ressources,
                 mode = 'final',
                 acquisition_function = 'EI',
                 runner_type='StandardRunner'):

# TODO if the same tuning run is done again but with e.g. different acquisition function the results folder is already filled with previous runs!
        # make sure to use another folder in this case
        hyperparams = [dic['name'] for dic in bounds]
        super(GP, self).__init__(optimizer_class, hyperparams, ressources, runner_type)

        self._acquisition_function = acquisition_function
        self._bounds = bounds
        self._mode = mode

    def _determine_cost_from_output_and_mode(self, output):
        # check if accuracy is available, else take loss
        # check must work for both frameworks
        if all(v==0 for v in output['test_accuracies']) or not output['test_accuracies']:
            cost = output['test_losses']
        else:
            # -1 because GPyOpt looks for minimum of cost function
            cost = [-1*v for v in output['test_accuracies']]

        if self._mode == 'final':
            cost = cost[-1]
        elif self._mode == 'best':
            cost = min(cost)
        else:
            raise NotImplementedError('''Mode not available for this tuning method. Please use final or best.''')
        return cost

    def _build_hyperparams_dict_from_proposal(self, proposal_array):
        hyperparams = {}
        for index, key in enumerate(self._hyperparams):
            # TODO why the workaround with the extra dimension of the proposal?
            hyperparams[key] = proposal_array[0][index]
        return hyperparams

    # TODO abstract this
    def _generate_cost_function(self, testproblem, **kwargs):
        '''Factory to create the cost function depending on the testproblem and kwargs.'''
        def _cost_function(proposal_array):
            hyperparams = self._build_hyperparams_dict_from_proposal(proposal_array)
            runner = self._runner(self._optimizer_class)
            output = runner.run(testproblem, hyperparams, **kwargs)
            cost = self._determine_cost_from_output_and_mode(output)
            return cost
        return _cost_function

    def tune(self, testproblems, num_parallel_samples = 1, num_cores = 1, evalu_type = 'sequential', random_seed = 42, **kwargs):
        self._set_seed(random_seed)
        # testproblems can also be only one testproblem
        if type(testproblems) == str:
            testproblems=testproblems.split()
        ops = {}
        for testproblem in testproblems:
            # TODO how to use different kernels
            cost_function = self._generate_cost_function(testproblem, **kwargs)
            # TODO how to deal with minimum amount of init samples?
            op = BO(f=cost_function, domain=self._bounds, acquisition_type=self._acquisition_function, batch_size=num_parallel_samples, num_cores=num_cores, evaluator_type=evalu_type, initial_design_numdata=5)
            op.run_optimization(max_iter=self._ressources)
            ops[testproblem] = op
        # returns a dict of the op objects
        return ops

class TPE(Tuner):
    pass