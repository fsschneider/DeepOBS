# -*- coding: utf-8 -*-
import abc
from .. import config
from numpy.random import seed as np_seed
import os
import datetime
from .tuner_utils import rerun_setting
import copy


class Tuner(abc.ABC):
    """The base class for all tuning methods in DeepOBS.
    Attributes:
    _optimizer_class: See argument optimizer_class
    _optimizer_name: The name of the optimizer class
    _hyperparameter_names: A nested dictionary that lists all hyperparameters of the optimizer,
    their type and their default values
    _resources: The number of evaluations the tuner is allowed to perform on each testproblem.
    _runner_type: The DeepOBS runner type that the tuner uses for evaluation.
        """
    def __init__(self,
                 optimizer_class,
                 hyperparam_names,
                 ressources,
                 runner_type='StandardRunner'):
        """Args:
            optimizer_class: The optimizer class of the optimizer that is run on
            the testproblems. For PyTorch this must be a subclass of torch.optim.Optimizer. For
            TensorFlow a subclass of tf.train.Optimizer.

            hyperparam_names (dict): A nested dictionary that lists all hyperparameters of the optimizer,
            their type and their default values (if they have any) in the form: {'<name>': {'type': <type>, 'default': <default value>}},
            e.g. for torch.optim.SGD with momentum:
            {'lr': {'type': float},
            'momentum': {'type': float, 'default': 0.99},
            'uses_nesterov': {'type': bool, 'default': False}}

            ressources (int): The number of evaluations the tuner is allowed to perform on each testproblem.
            runner_type (str): The DeepOBS runner type that the tuner uses for evaluation.
        """

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

    @staticmethod
    def _set_seed(random_seed):
        """Sets all relevant seeds for the tuning."""
        np_seed(random_seed)

    def tune_on_testset(self, testset, *args, **kwargs):
        for testproblem in testset:
            self.tune(testproblem, *args, **kwargs)

    @abc.abstractmethod
    def tune(self, testproblem, *args, output_dir='./results', random_seed=42, rerun_best_setting = False, **kwargs):
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

    def _formate_hyperparam_names_to_string(self):
        str_dict = copy.deepcopy(self._hyperparam_names)
        for hp in self._hyperparam_names:
            str_dict[hp]['type'] = self._hyperparam_names[hp]['type'].__name__
        return str(str_dict)

    def _generate_python_script(self, generation_dir):
        if not os.path.isdir(generation_dir):
            os.makedirs(generation_dir, exist_ok=True)
        script = open(os.path.join(generation_dir, self._optimizer_name + '.py'), 'w')
        import_line1 = 'from deepobs.' + config.get_framework() + '.runners.runner import ' + self._runner_type
        import_line2 = 'from ' + self._optimizer_class.__module__ + ' import ' + self._optimizer_class.__name__
        script.write(import_line1 +
                     '\n' +
                     import_line2 +
                     '\nrunner = ' +
                     self._runner_type +
                     '(' +
                     self._optimizer_class.__name__ + ', '
                     + self._formate_hyperparam_names_to_string() +
                     ')\nrunner.run()')
        script.close()
        return self._optimizer_name + '.py'

    def _generate_hyperparams_formate_for_command_line(self, hyperparams):
        string = ''
        for key, value in hyperparams.items():
            if self._hyperparam_names[key]['type'] == bool:
                string += ' --' + key
            else:
                string += ' --' + key + ' ' + str(value)
        return string

    @staticmethod
    # TODO how to deal with the training_params dict?
    def _generate_kwargs_format_for_command_line(**kwargs):
        string = ''
        for key, value in kwargs.items():
            string += ' --' + key + ' ' + str(value)
        return string

    def tune(self, testproblem, output_dir='./results', random_seed=42, rerun_best_setting = False, **kwargs):
        self._set_seed(random_seed)
        params = self._sample()
        for sample in params:
            runner = self._runner(self._optimizer_class, self._hyperparam_names)
            runner.run(testproblem, hyperparams=sample, random_seed=random_seed, output_dir=output_dir, **kwargs)

        if rerun_best_setting:
            optimizer_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            rerun_setting(self._runner, self._optimizer_class, self._hyperparam_names, optimizer_path)

    def generate_commands_script(self, testproblem, output_dir='./results', random_seed=42, generation_dir = './command_scripts', **kwargs):
        script = self._generate_python_script(generation_dir)
        file = open(os.path.join(generation_dir, 'jobs_' + self._optimizer_name + '_' + self._search_name + '_' + testproblem + '.txt'), 'w')
        kwargs_string = self._generate_kwargs_format_for_command_line(**kwargs)
        self._set_seed(random_seed)
        params = self._sample()
        for sample in params:
            sample_string = self._generate_hyperparams_formate_for_command_line(sample)
            file.write('python3 ' + script + ' ' + testproblem + ' ' + sample_string + ' --random_seed ' + str(
                random_seed) + ' --output_dir' + output_dir + ' ' + kwargs_string + '\n')
        file.close()

    def generate_commands_script_for_testset(self, testset, *args, **kwargs):
        for testproblem in testset:
            self.generate_commands_script(testproblem, *args, **kwargs)
