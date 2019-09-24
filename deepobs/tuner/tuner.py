# -*- coding: utf-8 -*-
import abc
from numpy.random import seed as np_seed
import os
from .tuner_utils import rerun_setting
from deepobs.analyzer.shared_utils import _dump_json


class Tuner(abc.ABC):
    """
    The base class for all tuning methods in DeepOBS.
    """

    def __init__(self,
                 optimizer_class,
                 hyperparam_names,
                 ressources,
                 runner):
        """
        Args:
            optimizer_class (framework optimizer class): The optimizer class of the optimizer that is run on \
            the testproblems. For PyTorch this must be a subclass of torch.optim.Optimizer. For \
            TensorFlow a subclass of tf.train.Optimizer.
            hyperparam_names (dict): A nested dictionary that lists all hyperparameters of the optimizer, \
            their type and their default values (if they have any) in the form: {'<name>': {'type': <type>, 'default': <default value>}}, \
            e.g. for torch.optim.SGD with momentum: \
            {'lr': {'type': float}, \
            'momentum': {'type': float, 'default': 0.99}, \
            'uses_nesterov': {'type': bool, 'default': False}}
            ressources (int): The number of evaluations the tuner is allowed to perform on each testproblem.
            runner: The DeepOBS runner that the tuner uses for evaluation.
        """

        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._hyperparam_names = hyperparam_names
        self._ressources = ressources
        self._runner = runner

    @staticmethod
    def _set_seed(random_seed):
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
    def tune(self, testproblem, *args, output_dir='./results', random_seed=42, rerun_best_setting = True, **kwargs):
        """Tunes hyperparaneter of the optimizer_class on a testproblem.
        Args:
            testproblem (str): Testproblem for which to generate commands.
            output_dir (str): The output path where the execution results are written to.
            random_seed (int): The random seed for the tuning.
            rerun_best_setting (bool): Whether to rerun the best setting with 10 different seeds.
        """
        pass


class ParallelizedTuner(Tuner):
    """
    The base class for all tuning methods which are uninformed and parallelizable, like Grid Search and Random Search.
    """
    def __init__(self,
                 optimizer_class,
                 hyperparam_names,
                 ressources,
                 runner):
        super(ParallelizedTuner, self).__init__(optimizer_class,
                                                hyperparam_names,
                                                ressources,
                                                runner)

    @abc.abstractmethod
    def _sample(self):
        return

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
        """Tunes the optimizer on the test problem.
        Args:
            testproblem (str): The test problem to tune the optimizer on.
            output_dir (str): The output directory for the results.
            random_seed (int): Random seed for the whole truning process. Every individual run is seeded by it.
            rerun_best_setting (bool): Whether to automatically rerun the best setting with 10 different seeds.
        """
        self._set_seed(random_seed)
        params = self._sample()
        for sample in params:
            runner = self._runner(self._optimizer_class, self._hyperparam_names)
            runner.run(testproblem, hyperparams=sample, random_seed=random_seed, output_dir=output_dir, **kwargs)

        if rerun_best_setting:
            optimizer_path = os.path.join(output_dir, testproblem, self._optimizer_name)
            rerun_setting(self._runner, self._optimizer_class, self._hyperparam_names, optimizer_path)

    def generate_commands_script(self, testproblem, run_script, output_dir='./results', random_seed=42,
                                 generation_dir = './command_scripts', **kwargs):
        """
        Args:
            testproblem (str): Testproblem for which to generate commands.
            run_script (str): Name the run script that is used from the command line.
            output_dir (str): The output path where the execution results are written to.
            random_seed (int): The random seed for the tuning.
            generation_dir (str): The path to the directory where the generated scripts are written to.

        Returns:
            str: The relative file path to the generated commands script.

        """

        os.makedirs(generation_dir, exist_ok=True)
        file_path = os.path.join(generation_dir, 'jobs_' + self._optimizer_name + '_' + self._search_name + '_' + testproblem + '.txt')
        file = open(file_path, 'w')
        kwargs_string = self._generate_kwargs_format_for_command_line(**kwargs)
        self._set_seed(random_seed)
        params = self._sample()
        for sample in params:
            sample_string = self._generate_hyperparams_format_for_command_line(sample)
            file.write('python3 ' + run_script + ' ' + testproblem + ' ' + sample_string + ' --random_seed ' + str(
                random_seed) + ' --output_dir ' + output_dir + ' ' + kwargs_string + '\n')
        file.close()
        return file_path

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
