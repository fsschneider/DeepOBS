# -*- coding: utf-8 -*-
from .tuner import Tuner
from .. import config
class GridSearch(Tuner):

    def __init__(self, optimizer_class, hyperparams, grid, ressources, mode = 'final', testproblems=None, runner_type='StandardRunner'):
        super(GridSearch).__init__(self, optimizer_class, hyperparams, ressources, mode, testproblems, runner_type)
        # TODO determine grid from ressources
        self._grid = grid
#        self._grid = self.__create_grid(self)

#    def __create_grid(self):
#        sample_size_per_parameter = self._ressources // len(self._hyperparams)
#        for name in self._hyperparams.iterkeys():

    def tune(self, testproblem, **training_params):
        # TODO parallelize
        # TODO return the commands if no parallelization possible
        for grid_point in self._grid:
            optimizer_settings = dict(zip(self._hyperparams, grid_point))
            runner = self._runner(self._optimizer_class, optimizer_settings)
            result = runner.run(testproblem, **config.get_testproblem_default_setting(testproblem), **training_params)
        return

    def generate_jobs_script(self, testproblem, **training_params):


