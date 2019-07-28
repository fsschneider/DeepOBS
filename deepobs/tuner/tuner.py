# -*- coding: utf-8 -*-
import abc
from .. import config
from numpy.random import seed as np_seed
import os
from .tuner_utils import rerun_setting
import copy


class Tuner(abc.ABC):
    """The base class for all tuning methods in DeepOBS.
    Attributes:
        _optimizer_class: See argument optimizer_class
        _optimizer_name: The name of the optimizer class
        _hyperparam_names: A nested dictionary that lists all hyperparameters of the optimizer,
        their type and their default values
        _ressources: The number of evaluations the tuner is allowed to perform on each testproblem.
        _runner_type: The DeepOBS runner type that the tuner uses for evaluation.
    Methods:
        _set_seed: Sets all random seeds.
        tune: Tunes the optimizer on a testproblem.
        tune_on_testset: Tunes the optimizer on each testproblem of a testset.
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
        """Tunes the hyperparameter on a whole testset.
        Args:
            testset (list): A list of testproblems.
        """
        if any(s in kwargs for s in ['num_epochs', 'batch_size', 'weight_decay']):
            raise RuntimeError('Cannot execute tuning on a whole testset if num_epochs, '
                               'weight_decay or batch_size is set. '
                               'A testset tuning is ment to tune on default testproblems.')
        for testproblem in testset:
            self.tune(testproblem, *args, **kwargs)

    @abc.abstractmethod
    def tune(self, testproblem, *args, output_dir='./results', random_seed=42, rerun_best_setting = False, **kwargs):
        """Tunes hyperparaneter of the optimizer_class on a testproblem.
        Args:
            testproblem (str): Testproblem for which to generate commands.
            output_dir (str): The output path where the execution results are written to.
            random_seed (int): The random seed for the tuning.
            rerun_best_setting (bool): Whether to rerun the best setting with 10 different seeds.
        """
        pass


class ParallelizedTuner(Tuner):
    """The base class for all tuning methods which are uninformed and parallelizable, like Grid Search and Random Search.
    Methods:
        _sample: Creates a list of all hyperparameter settings that are to evaluate.
        generate_commands_script: Generates commands to allow the user to execute each tuning job seperately.
        generate_commands_script_for_testset: Generates commands for each testproblem in a testset.
    """
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

    def __formate_hyperparam_names_to_string(self):
        str_dict = copy.deepcopy(self._hyperparam_names)
        for hp in self._hyperparam_names:
            str_dict[hp]['type'] = self._hyperparam_names[hp]['type'].__name__
        return str(str_dict)

    def __generate_python_script(self, generation_dir):
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
                     + self.__formate_hyperparam_names_to_string() +
                     ')\nrunner.run()')
        script.close()
        return self._optimizer_name + '.py'

    def _generate_hyperparams_format_for_command_line(self, hyperparams):
        """Overwrite this method to specify how hyperparams should be represented in the command line string.
        This is basically the inversion of your runner specific method ``_add_hyperparams_to_argparse``"""
        string = ''
        for key, value in hyperparams.items():
            if self._hyperparam_names[key]['type'] == bool:
                string += ' --' + key
            else:
                string += ' --' + key + ' ' + str(value)
        return string

    def _generate_kwargs_format_for_command_line(self, **kwargs):
        """Overwrite this method to specify how additional training params should be represented in the command line string.
        This is basically the inversion of your runner specific method ``_add_training_params_to_argparse``"""
        string = ''
        for key, value in kwargs.items():
            if key == 'lr_sched_factors' or key == 'lr_sched_epochs':
                string += ' --' + key
                for v in value:
                    string += ' ' + str(v)
            else:
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

    def generate_commands_script(self, testproblem, output_dir='./results', random_seed=42,
                                 generation_dir = './command_scripts', **kwargs):
        """
        Args:
            testproblem (str): Testproblem for which to generate commands.
            output_dir (str): The output path where the execution results are written to.
            random_seed (int): The random seed for the tuning.
            generation_dir (str): The path to the directory where the generated scripts are written to.

        """

        script = self.__generate_python_script(generation_dir)
        file = open(os.path.join(generation_dir, 'jobs_' + self._optimizer_name + '_' + self._search_name + '_' + testproblem + '.txt'), 'w')
        kwargs_string = self._generate_kwargs_format_for_command_line(**kwargs)
        self._set_seed(random_seed)
        params = self._sample()
        for sample in params:
            sample_string = self._generate_hyperparams_format_for_command_line(sample)
            file.write('python3 ' + script + ' ' + testproblem + ' ' + sample_string + ' --random_seed ' + str(
                random_seed) + ' --output_dir ' + output_dir + ' ' + kwargs_string + '\n')
        file.close()

    def generate_commands_script_for_testset(self, testset, *args, **kwargs):
        """Generates command scripts for a whole testset.
        Args:
            testset (list): A list of the testproblem strings.
            """
        if any(s in kwargs for s in ['num_epochs', 'batch_size', 'weight_decay']):
            raise RuntimeError('Cannot execute tuning on a whole testset if num_epochs, '
                               'weight_decay or batch_size is set. '
                               'A testset tuning is ment to tune on default testproblems.')
        for testproblem in testset:
            self.generate_commands_script(testproblem, *args, **kwargs)
