# -*- coding: utf-8 -*-
from .tuner import ParallelizedTuner
from itertools import product


class GridSearch(ParallelizedTuner):
    """
    A basic Grid Search tuner.
    """
    def __init__(self, optimizer_class, hyperparam_names, grid, ressources, runner):
        """
        Args:
            grid (dict): Holds the discrete values for each hyperparameter as lists.
        """
        super(GridSearch, self).__init__(optimizer_class, hyperparam_names, ressources, runner)
        self.__check_if_grid_is_valid(grid, ressources)
        self._grid = grid
        self._search_name = 'grid_search'

    @staticmethod
    def __check_if_grid_is_valid(grid, ressources):
        grid_size = len(list(product(*[values for values in grid.values()])))
        if grid_size > ressources:
            raise RuntimeError('Grid is too large for the available number of iterations.')

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
