import json
import os
import warnings

import numpy as np


def _check_setting_folder_is_not_empty(setting_path):
    runs = [run for run in os.listdir(setting_path) if 'json' in run]
    try:
        assert len(runs) > 0
    except AssertionError:
        print('Found a setting folder with no runs inside: {0:s}'.format(
            setting_path))


def _check_output_structure(path, file_name):
    """Checks whether the output json is valid"""
    json_data = _load_json(path, file_name)
    try:
        # meta data must be in output
        assert 'num_epochs' in json_data
        assert 'batch_size' in json_data
        assert 'testproblem' in json_data
        assert 'random_seed' in json_data
        assert 'optimizer_name' in json_data
        assert 'optimizer_hyperparams' in json_data

        # must contain at least losses
        assert 'train_losses' in json_data
        assert 'valid_losses' in json_data
        assert 'test_losses' in json_data

        # all must have the same length
        assert len(json_data['train_losses']) == len(
            json_data['test_losses']) == len(
                json_data['valid_losses']) == json_data['num_epochs'] + 1
    except AssertionError as e:
        print('Found corrupted output file: {0:s} in path: {1:s}'.format(
            file_name, path))


def aggregate_runs(setting_folder, custom_metrics=None):
    """Aggregates all seed runs for a setting.
    Args:
        setting_folder (str): The path to the setting folder.
        custom_metrics (list(str)): Additional metrics that will be extracted if available
    Returns:
        A dictionary that contains the aggregated mean and std of all metrices, as well as the meta data.
        """
    dobs_metrics = [
        'train_losses', 'valid_losses', 'test_losses', 'train_accuracies',
        'valid_accuracies', 'test_accuracies'
    ]
    if custom_metrics is None:
        custom_metrics = []

    runs = [run for run in os.listdir(setting_folder) if run.endswith(".json")]

    def no_data():
        return []

    all_metrics = dobs_metrics + custom_metrics
    all_metrics_data = {m: no_data() for m in all_metrics}

    for run in runs:
        json_data = _load_json(setting_folder, run)
        for metric in all_metrics:
            try:
                run_data = json_data[metric]
            except KeyError:
                run_data = no_data()
            all_metrics_data[metric].append(run_data)

    # custom metrics: fill with nans if run quit earlier
    metrics_require_nans = set()
    nans_inserted = 0
    for metric in custom_metrics:
        max_num_points = max(
            len(run_data) for run_data in all_metrics_data[metric])
        # fill up with nans
        for run_data in all_metrics_data[metric]:
            while len(run_data) < max_num_points:
                metrics_require_nans.add(metric)
                nans_inserted += 1
                run_data.append(float('nan'))
    if nans_inserted > 0:
        print(
            "[CUSTOM METRICS]: Needed to insert {} NaNs".format(nans_inserted))
        print("[CUSTOM METRICS]: Affected metrics {}".format(
            metrics_require_nans))

    aggregate = dict()
    for metric in all_metrics:
        data = np.array(all_metrics_data[metric])
        # only add the metric if available
        is_empty = data.shape[1] == 0
        if not is_empty:
            aggregate[metric] = {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'all_final_values': [met[-1] for met in data],
                'lower_quartile': np.quantile(data, 0.25, axis=0),
                'median': np.median(data, axis=0),
                'upper_quartile': np.quantile(data, 0.75, axis=0),
                'mean_log': np.power(10, np.mean(np.log10(data), axis=0)),
                'std_log': np.power(10, np.std(np.log10(data), axis=0)),
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0),
            }
    # merge meta data
    aggregate['optimizer_hyperparams'] = json_data['optimizer_hyperparams']
    aggregate['training_params'] = json_data['training_params']
    aggregate['testproblem'] = json_data['testproblem']
    aggregate['num_epochs'] = json_data['num_epochs']
    aggregate['batch_size'] = json_data['batch_size']
    return aggregate


def _read_all_settings_folders(optimizer_path):
    """Returns a list of all setting folders in ``optimizer_path``"""
    optimizer_path = os.path.join(optimizer_path)
    return [
        f for f in os.listdir(optimizer_path)
        if os.path.isdir(os.path.join(optimizer_path, f)) and 'num_epochs' in f
    ]


def _check_if_metric_is_available(optimizer_path, metric):
    """Checks if the metric ``metric`` is availabe for the runs in ``optimizer_path``"""
    settings = _read_all_settings_folders(optimizer_path)
    sett = settings[0]
    path = os.path.join(optimizer_path, sett)
    run = [r for r in os.listdir(path) if '.json' in r][0]
    json_data = _load_json(path, run)
    if metric in json_data:
        return True
    else:
        return False


def _determine_available_metric(optimizer_path,
                                metric,
                                default_metric='valid_losses'):
    """Checks if the metric ``metric`` is availabe for the runs in ``optimizer_path``.
    If not, it returns the fallback metric ``default_metric``."""
    optimizer_name, testproblem_name = _get_optimizer_name_and_testproblem_from_path(
        optimizer_path)
    if _check_if_metric_is_available(optimizer_path, metric):
        return metric
    else:

        # TODO remove if-else once validation metrics are available for the baselines
        if _check_if_metric_is_available(optimizer_path, default_metric):
            warnings.warn(
                'Metric {0:s} does not exist for testproblem {1:s}. We now use fallback metric {2:s}'
                .format(metric, testproblem_name,
                        default_metric), RuntimeWarning)
            return default_metric
        else:
            warnings.warn(
                'Cannot fallback to metric {0:s} for optimizer {1:s} on testproblem {2:s}. Will now fallback to metric test_losses'
                .format(default_metric, optimizer_name,
                        testproblem_name), RuntimeWarning)
            return 'test_losses'


def _dump_json(path, file, obj):
    with open(os.path.join(path, file), 'w') as f:
        f.write(json.dumps(obj))


def _append_json(path, file, obj):
    with open(os.path.join(path, file), 'a') as f:
        f.write(json.dumps(obj))
        f.write('\n')


def _clear_json(path, file):
    json_path = os.path.join(path, file)
    if os.path.exists(json_path):
        os.remove(json_path)


def _load_json(path, file_name):
    with open(os.path.join(path, file_name), "r") as f:
        json_data = json.load(f)
    return json_data


def _get_all_setting_analyzer(optimizer_path, custom_metrics=None):
    """Creates a list of SettingAnalyzers (one for each setting in ``optimizer_path``)"""
    if custom_metrics is None:
        custom_metrics = []

    optimizer_path = os.path.join(optimizer_path)
    setting_folders = _read_all_settings_folders(optimizer_path)
    setting_analyzers = []
    for sett in setting_folders:
        sett_path = os.path.join(optimizer_path, sett)
        setting_analyzers.append(
            SettingAnalyzer(sett_path, custom_metrics=custom_metrics))
    return setting_analyzers


def _get_optimizer_name_and_testproblem_from_path(optimizer_path):
    optimizer_name = os.path.split(optimizer_path)[-1]
    testproblem = os.path.split(os.path.split(optimizer_path)[0])[-1]
    return optimizer_name, testproblem


def create_setting_analyzer_ranking(optimizer_path,
                                    mode='final',
                                    metric='valid_accuracies',
                                    custom_metrics=None):
    """Reads in all settings in ``optimizer_path`` and sets up a ranking by returning an ordered list of SettingAnalyzers.
    Args:
        optimizer_path (str): The path to the optimizer to analyse.
        mode (str): The mode by which to decide the best setting.
        metric (str): The metric by which to decide the best setting.
    Returns:
        An ordered list of SettingAnalyzers. I.e. the first item is considered 'the best one' etc.
    """
    if custom_metrics is None:
        custom_metrics = []

    metric = _determine_available_metric(optimizer_path, metric)
    setting_analyzers = _get_all_setting_analyzer(
        optimizer_path, custom_metrics=custom_metrics)

    if 'acc' in metric:
        sgn = -1
    else:
        sgn = 1

    if mode == 'final':
        setting_analyzers_ordered = sorted(
            setting_analyzers,
            key=lambda idx: sgn * idx.get_final_value(metric))
    elif mode == 'best':
        setting_analyzers_ordered = sorted(
            setting_analyzers,
            key=lambda idx: sgn * idx.get_best_value(metric))
    elif mode == 'most':
        # if all have the same amount of runs, i.e. no 'most' avalaible, fall back to 'final'
        if all(x.n_runs == setting_analyzers[0].n_runs
               for x in setting_analyzers):
            optimizer_name, testproblem_name = _get_optimizer_name_and_testproblem_from_path(
                optimizer_path)
            warnings.warn(
                'All settings for {0:s} on test problem {1:s} have the same number of seeds runs. Mode \'most\' does not make sense and we use the fallback mode \'final\''
                .format(optimizer_path, testproblem_name), RuntimeWarning)
            setting_analyzers_ordered = sorted(
                setting_analyzers,
                key=lambda idx: sgn * idx.get_final_value(metric))
        else:
            setting_analyzers_ordered = sorted(setting_analyzers,
                                               key=lambda idx: idx.n_runs,
                                               reverse=True)
    else:
        raise RuntimeError('Mode not implemented')

    return setting_analyzers_ordered


class SettingAnalyzer:
    """DeepOBS analyzer class for a setting (a hyperparameter setting).

    Attributes:
        path (str): Path to the setting folder.
        aggregate (dictionary): Contains the mean and std of the runs as well as the meta data.
        n_runs (int): The number of seed runs that were performed for this setting.
    """
    def __init__(self, path, custom_metrics=None):
        """Initializes a new SettingAnalyzer instance.

        Args:
            path (str): String to the setting folder.
        """
        if custom_metrics is None:
            custom_metrics = []

        self.path = path
        self.n_runs = self.__get_number_of_runs()
        self.aggregate = aggregate_runs(path, custom_metrics=custom_metrics)

    def __get_number_of_runs(self):
        """Calculates the total number of seed runs."""
        return len(
            [run for run in os.listdir(self.path) if run.endswith(".json")])

    def get_final_value(self, metric):
        """Get the final (mean) value of the metric."""
        try:
            return self.aggregate[metric]['mean'][-1]
        except KeyError:
            raise KeyError(
                'Metric {0:s} not available for testproblem {1:s} of this setting'
                .format(metric, self.aggregate['testproblem']))

    def get_best_value(self, metric):
        """Get the best (mean) value of the metric."""
        try:
            if 'loss' in metric:
                return min(self.aggregate[metric]['mean'])
            elif 'acc' in metric:
                return max(self.aggregate[metric]['mean'])
            else:
                raise NotImplementedError
        except KeyError:
            raise KeyError(
                'Metric {0:s} not available for testproblem {1:s} of this setting'
                .format(metric, self.aggregate['testproblem']))

    def calculate_speed(self, conv_perf_file):
        """Calculates the speed of the setting."""
        path, file = os.path.split(conv_perf_file)
        conv_perf = _load_json(path, file)[self.aggregate['testproblem']]

        runs = [run for run in os.listdir(self.path) if run.endswith(".json")]
        metric = 'test_accuracies' if 'test_accuracies' in self.aggregate else 'test_losses'
        perf_values = []

        for run in runs:
            json_data = _load_json(self.path, run)
            perf_values.append(json_data[metric])

        perf_values = np.array(perf_values)
        if metric == "test_losses":
            # average over first time they reach conv perf (use num_epochs if conv perf is not reached)
            speed = np.mean(
                np.argmax(perf_values <= conv_perf, axis=1) +
                np.invert(np.max(perf_values <= conv_perf, axis=1)) *
                perf_values.shape[1])
        elif metric == "test_accuracies":
            speed = np.mean(
                np.argmax(perf_values >= conv_perf, axis=1) +
                np.invert(np.max(perf_values >= conv_perf, axis=1)) *
                perf_values.shape[1])
        else:
            raise NotImplementedError

        return speed

    def get_all_final_values(self, metric):
        """Get all final values of the seed runs for the metric."""
        try:
            return self.aggregate[metric]['all_final_values']
        except KeyError:
            raise KeyError(
                'Metric {0:s} not available for testproblem {1:s} of this setting'
                .format(metric, self.aggregate['testproblem']))
