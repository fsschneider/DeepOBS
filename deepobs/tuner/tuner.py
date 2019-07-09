# -*- coding: utf-8 -*-
import abc
from .. import config
from numpy.random import seed as np_seed
from .tuner_utils import create_tuning_ranking
from .tuner_utils import _load_json
import os
import numpy as np


class Tuner(abc.ABC):
    def __init__(self,
                 optimizer_class,
                 hyperparam_names,
                 ressources,
                 runner_type='StandardRunner'):

        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._hyperparam_names = hyperparam_names
        self._ressources = ressources
        self._runner_type = runner_type

        if config.get_framework() == 'tensorflow':
            from .. import tensorflow as fw
        elif config.get_framework() == 'pytorch':
            from .. import pytorch as fw
        else:
            raise RuntimeError('Framework not implemented.')
        # check if requested runner is implemented as a class
        try:
            self._runner = getattr(fw.runners.runner, runner_type)
        except AttributeError:
            raise AttributeError('Runner type ', runner_type,
                                 ' not implemented. If you really need it, you have to implement it on your own.')

    def rerun_best_setting(self, optimizer_path, seeds = np.arange(43, 52), rank=1, mode='final', aggregated=False):
        optimizer_path = os.path.join(optimizer_path)
        # TODO would be easier if that also returns the path to the setting
        ranking = create_tuning_ranking(optimizer_path, mode, aggregated)
        setting = ranking[rank - 1]['parameters']
        runner = self._runner(self._optimizer_class, self._hyperparam_names)

        # read in meta data, such as num_epochs, testproblem, etc
        a_sett_path = [sett for sett in os.listdir(optimizer_path) if os.path.isdir(os.path.join(optimizer_path, sett))][0]
        a_json = [j for j in os.listdir(os.path.join(optimizer_path, a_sett_path))][0]
        json_data = _load_json(a_sett_path, a_json)
        testproblem = json_data['testproblem']
        num_epochs = json_data['num_epochs']
        batch_size = json_data['batch_size']

        results_path = os.path.split(os.path.split(optimizer_path)[0])[0]
        for seed in seeds:
            runner.run(testproblem, hyperparams = setting, random_seed = int(seed), num_epochs = num_epochs, batch_size = batch_size, output_dir = results_path)


    @staticmethod
    def _set_seed(random_seed):
        np_seed(random_seed)

    @staticmethod
    def _check_output_path(path):
        """Checks if path already exists. creates it if not, cleans it if yes."""
        # TODO warn user that path will be cleaned up! data might be lost for users if they dont know
        # TODO or add some unique id to the outdir?
        if not os.path.isdir(path):
            # create if it does not exist
            os.makedirs(path)
            # TODO I dont delete the folder for now!! not a good idea (e.g. momentum = SGD in torch)

    #        else:
    #            # delete content if it exist
    #            contentlist = os.listdir(path)
    #            for f in contentlist:
    #                _path = os.path.join(path, f)
    #                if os.path.isfile(_path):
    #                    os.remove(_path)
    #                elif os.path.isdir(_path):
    #                    shutil.rmtree(_path)

    @staticmethod
    def _read_testproblems(testproblems):
        if type(testproblems) == str:
            testproblems = testproblems.split()
        else:
            testproblems = sorted(testproblems)
        return testproblems

    @abc.abstractmethod
    def tune(self):
        pass


class ParallelizedTuner(Tuner):
    def __init__(self,
                 optimizer_class,
                 hyperparam_names,
                 ressources,
                 runner_type='StandardRunner'):
        super(ParallelizedTuner, self).__init__(optimizer_class,
                                                hyperparam_names,
                                                ressources,
                                                runner_type)

    @abc.abstractmethod
    def _sample(self):
        return

    # TODO hyperparam_names['type'] is formatted incorrectly
    # TODO smarter way to create that file?
    def _generate_python_script(self):
        script = open(self._optimizer_name + '.py', 'w')
        # TODO vereinheitliche runner paths
        import_line1 = 'from deepobs.' + config.get_framework() + '.runners.runner import ' + self._runner_type
        import_line2 = 'from ' + self._optimizer_class.__module__ + ' import ' + self._optimizer_class.__name__
        # TODO optimizer_class  must implement __module__ and __name__ accordingly
        script.write(import_line1 +
                     '\n' +
                     import_line2 +
                     '\nrunner = ' +
                     self._runner_type +
                     '(' +
                     self._optimizer_class.__name__ + ', '
                     + str(self._hyperparam_names) +
                     ')\nrunner.run()')
        script.close()
        return self._optimizer_name + '.py'

    @staticmethod
    def _generate_hyperparams_formate_for_command_line(hyperparams):
        string = ''
        for key, value in hyperparams.items():
            string += ' --' + key + ' ' + str(value)
        return string

    @staticmethod
    # TODO how to deal with the training_params dict?
    def _generate_kwargs_format_for_command_line(**kwargs):
        string = ''
        for key, value in kwargs.items():
            string += ' --' + key + ' ' + str(value)
        return string

    def tune(self, testproblems, output_dir='./results', random_seed=42, rerun_best_setting = False, **kwargs):
        testproblems = self._read_testproblems(testproblems)
        for testproblem in testproblems:
            self._set_seed(random_seed)
            log_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            self._check_output_path(log_path)

            # TODO better prints or not at all
            params = self._sample()
            print('Tuning', self._optimizer_name, 'on testproblem', testproblem)
            for sample in params:
                print('Start training with', sample)
                runner = self._runner(self._optimizer_class, self._hyperparam_names)
                runner.run(testproblem, hyperparams=sample, random_seed=random_seed, output_dir=output_dir, **kwargs)
            if rerun_best_setting:
                # TODO momentum is rerun in SGD folder!!
                optimizer_path = os.path.join(output_dir, testproblem, self._optimizer_name)
                self.rerun_best_setting(optimizer_path)

    # TODO write into subfolder
    def generate_commands_script(self, testproblems, output_dir='./results', random_seed=42, **kwargs):
        testproblems = self._read_testproblems(testproblems)
        script = self._generate_python_script()
        file = open('jobs_' + self._optimizer_name + '_' + self._search_name + '.txt', 'w')
        kwargs_string = self._generate_kwargs_format_for_command_line(**kwargs)
        for testproblem in testproblems:
            self._set_seed(random_seed)
            log_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            self._check_output_path(log_path)
            params = self._sample()
            file.write('##### ' + testproblem + ' #####\n')
            for sample in params:
                sample_string = self._generate_hyperparams_formate_for_command_line(sample)
                file.write('python3 ' + script + ' ' + testproblem + ' ' + sample_string + ' ' + '--random_seed ' + str(
                    random_seed) + '--output_dir' + output_dir + ' ' + kwargs_string + '\n')
        file.close()
