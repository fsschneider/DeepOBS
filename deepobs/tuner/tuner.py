# -*- coding: utf-8 -*-
import abc
from .. import config
from numpy.random import seed as np_seed
import os
import json

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

    def _init_tuning_summary(self):
        pass
    def _write_tuning_summary(self, step, testproblem, output_dir, runner_output):
        path = os.path.join(output_dir, testproblem, self._optimizer_name)
        path += 'tuner_log.json'
        summary_dict['final_test_loss'] = runner_output['test_losses'][-1]
        # TODO this will not work for tensorflow where acc might be empty
        # TODO this is one reason more to unify the runner outputs
        summary_dict['final_test_accuracy'] = runner_output['test_accuracies'][-1]
        summary_dict['optimizer_hyperparams'] = runner_output['optimizer_hyperparams']
        summary_dict['testproblem'] = runner_output['testproblem']
        summary_dict['optimizer'] = runner_output['optimizer']
    
        with open(path, 'r') as f:
            json_dict = f.load(path)
        
        with open() as f:
            f.write(json.dumps(summary_dict))
            
    # TODO add output dir to command line string
    def tune(self, testproblems, output_dir = './results', random_seed=42, **kwargs):
        # testproblems can also be only one testproblem
        self._set_seed(random_seed)
        if type(testproblems) == str:
            testproblems=testproblems.split()
        for testproblem in testproblems:
            params = self._sample()
            print('Tuning', self._optimizer_name, 'on testproblem', testproblem)
            for sample in params:
                print('Start training with', sample)
                runner = self._runner(self._optimizer_class)
                runner.run(testproblem, hyperparams=sample, random_seed=random_seed, output_dir = output_dir, **kwargs)
                

        
# TODO write into subfolder
    def generate_commands_script(self, testproblems, random_seed = 42, **kwargs):
        # TODO rather seed in testproblems loop? otherwise order of testproblems changes the seeds for each of them
        self._set_seed(random_seed)
        # testproblems can also be only one testproblem
        if type(testproblems) == str:
            testproblems=testproblems.split()
        script = self._generate_python_script()
        file = open('jobs_'+ self._optimizer_name  + '_' + self._search_name + '.txt', 'w')
        kwargs_string = self._generate_kwargs_format_for_command_line(**kwargs)
        for testproblem in testproblems:
            params = self._sample()
            file.write('##### ' + testproblem + ' #####\n')
            for sample in params:
                sample_string = self._generate_hyperparams_formate_for_command_line(sample)
                file.write('python3 ' + script + ' ' + testproblem + ' ' + sample_string + ' ' + '--random_seed ' + str(random_seed) + ' ' + kwargs_string  + '\n')
        file.close()