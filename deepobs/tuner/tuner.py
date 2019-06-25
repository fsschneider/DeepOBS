# -*- coding: utf-8 -*-
import abc
from .. import config
from numpy.random import seed as np_seed
import os
import shutil

class Tuner(abc.ABC):
    def __init__(self,
                 optimizer_class,
                 hyperparams,
                 ressources,
                 runner_type = 'StandardRunner'):

        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._hyperparams = hyperparams
        self._ressources = ressources
        self._runner_type = runner_type

    # where to make framework setable by the user?
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
            raise AttributeError('Runner type ', runner_type,' not implemented. If you really need it, you have to implement it on your own.')

    @staticmethod
    def _set_seed(random_seed):
        # TODO which other seeds to include?
        np_seed(random_seed)
    
    @staticmethod
    def _check_output_path(path):
        """Checks if path already exists. creates it if not, cleans it if yes."""
        # TODO warn user that path will be cleaned up! data might be lost for users if they dont know
        # TODO or maybe add some unique id to the outdir?
        if not os.path.isdir(path):
            # create if it does not exist
            os.makedirs(path)
        else:
            # delete content if it exist
            contentlist = os.listdir(path)
            for f in contentlist:
                _path = os.path.join(path, f)
                if os.path.isfile(_path):
                    os.remove(_path)
                elif os.path.isdir(_path):
                    shutil.rmtree(_path)
    
    @staticmethod
    def _read_testproblems(testproblems):
        if type(testproblems) == str:
            testproblems=testproblems.split()
        else:
            testproblems = sorted(testproblems)
        return testproblems
    
    @abc.abstractmethod
    def tune():
        pass
        
class ParallelizedTuner(Tuner):
    def __init__(self,
                 optimizer_class,
                 hyperparams,
                 ressources,
                 runner_type = 'StandardRunner'):
        super(ParallelizedTuner, self).__init__(optimizer_class,
                                                hyperparams,
                                                ressources,
                                                runner_type)
    @abc.abstractmethod
    def _sample(self):
        return

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
                     self._optimizer_class.__name__ +
                     ')\nrunner.run()')
        script.close()
        return self._optimizer_name + '.py'

    @staticmethod
    def _generate_hyperparams_formate_for_command_line(hyperparams):
        string = ''
        for key,value in hyperparams.items():
            string = string + key + '=' + str(value) + ',,'
        string = string[:-2]
        return string

    @staticmethod
    def _generate_kwargs_format_for_command_line(**kwargs):
        string = ''
        for key, value in kwargs.items():
            string += '--' + key + ' ' + str(value) + ' '
        string = string [:-1]
        return string
    
    # TODO add output dir to command line string
    def tune(self, testproblems, output_dir = './results', random_seed=42, **kwargs):
        self._set_seed(random_seed)
        testproblems = self._read_testproblems(testproblems)
        for testproblem in testproblems:
            log_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            self._check_output_path(log_path)
            
            params = self._sample()
            print('Tuning', self._optimizer_name, 'on testproblem', testproblem)
            for sample in params:
                print('Start training with', sample)
                runner = self._runner(self._optimizer_class)
                runner.run(testproblem, hyperparams=sample, random_seed=random_seed, output_dir = output_dir, **kwargs)
                
# TODO write into subfolder
    def generate_commands_script(self, testproblems, output_dir = './results', random_seed = 42, **kwargs):
        # TODO rather seed in testproblems loop? otherwise order of testproblems changes the seeds for each of them
        self._set_seed(random_seed)
        testproblems = self._read_testproblems(testproblems)
        script = self._generate_python_script()
        file = open('jobs_'+ self._optimizer_name  + '_' + self._search_name + '.txt', 'w')
        kwargs_string = self._generate_kwargs_format_for_command_line(**kwargs)
        for testproblem in testproblems:
            log_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            self._check_output_path(log_path)
            params = self._sample()
            file.write('##### ' + testproblem + ' #####\n')
            for sample in params:
                sample_string = self._generate_hyperparams_formate_for_command_line(sample)
                file.write('python3 ' + script + ' ' + testproblem + ' ' + sample_string + ' ' + '--random_seed ' + str(random_seed) + '--output_dir' + output_dir + ' ' + kwargs_string  + '\n')
        file.close()