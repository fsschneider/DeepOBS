# -*- coding: utf-8 -*-
from .tuner import ParallelizedTuner
from itertools import product

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class GridSearch(ParallelizedTuner):

    def __init__(self, optimizer_class, grid, ressources, runner_type='StandardRunner'):
        hyperparams = list(grid.keys())
        super(GridSearch, self).__init__(optimizer_class, hyperparams, ressources, runner_type)
        self._check_if_grid_is_valid(grid, ressources)
        self._grid = grid
        self._search_name = 'grid_search'

    @staticmethod
    def _check_if_grid_is_valid(grid, ressources):
        grid_size = len(list(product(*[values for values in grid.values()])))
        if grid_size > ressources:
            raise InputError('Grid is too large for the available number of iterations.')

    def _sample(self):
        all_values = []
        all_keys = []
        for key, values in self._grid.items():
            all_values.append(values)
            all_keys.append(key)

        samples = []
        for sample in product(*all_values):
            sample_dict = {}
            for index, value in enumerate(sample):
                sample_dict[all_keys[index]] = value
            samples.append(sample_dict)

        return samples