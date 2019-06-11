# -*- coding: utf-8 -*-
import abc
from .. import config
import os

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
            # TODO make sure that tf and pt pathes are consistent
            self._runner = getattr(fw.runners.runner, runner_type)
        except AttributeError:
            raise AttributeError('Runner type ', runner_type,' not implemented. If you really need it, you have to implement it on your own.')

# TODO testset automation
#    def tune_small_test_set(self, **training_params):
#        testproblems = config.get_small_test_set()
#        for testproblem in testproblems:
#            self.tune(testproblem, **training_params)
#
#    def tune_large_test_set(self, **training_params):
#        testproblems = config.get_large_test_set()
#        for testproblem in testproblems:
#            self.tune(testproblem, **training_params)

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

    def __create_folder(self):
        if not os.path.exists('./scripts'):
            os.makedirs('./scripts')
        return

    # TODO smarter way to create that file?
    def _generate_python_script(self):
        # TODO what happens if this file does exist already?
        script = open(self._optimizer_name + '.py', 'w')
        # TODO vereinheitliche runner paths
        import_line1 = 'from deepobs.' + config.get_framework() + '.runners.runner import ' + self._runner_type
        import_line2 = 'from ' + self._optimizer_class.__module__ + ' import ' + self._optimizer_class.__name__
        # TODO optimizer_class  must implement __str__ accordingly
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

# TODO write convenience method

# TODO write into subfolder
# TODO write different testproblems in different files
    def generate_commands_script(self, testproblems):
        script = self._generate_python_script()
        params = self._sample()
        file = open('jobs_'+ self._optimizer_name  + '_' + self._search_name + '.txt', 'w')
        for testproblem in testproblems:
            file.write('##### ' + testproblem + ' #####\n')
            for sample in params:
                sample_string = self._generate_hyperparams_formate_for_command_line(sample)
                file.write('python3 ' + script + ' ' + testproblem + ' ' + sample_string + '\n')
        file.close()