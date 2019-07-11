import json
import os
import numpy as np


def aggregate_runs(setting_folder):
    runs = [run for run in os.listdir(setting_folder) if run.endswith(".json")]
    # metrices
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for run in runs:
        json_data = _load_json(setting_folder, run)
        train_losses.append(json_data['train_losses'])
        test_losses.append(json_data['test_losses'])
        # just add accuracies to the aggregate if they are available
        if 'train_accuracies' in json_data :
            train_accuracies.append(json_data['train_accuracies'])
            test_accuracies.append(json_data['test_accuracies'])

    aggregate = dict()
    for metrics in ['train_losses', 'test_losses', 'train_accuracies', 'test_accuracies']:
        # only add the metric if available
        if len(eval(metrics)) != 0:
            aggregate[metrics] = {
                    'mean': np.mean(eval(metrics), axis=0),
                    'std': np.std(eval(metrics), axis=0)
                }
    # merge meta data
    aggregate['optimizer_hyperparams'] = json_data['optimizer_hyperparams']
    aggregate['testproblem'] = json_data['testproblem']
    aggregate['num_epochs'] = json_data['num_epochs']
    aggregate['batch_size'] = json_data['batch_size']
    return aggregate


def _read_all_settings_folders(optimizer_path):
    optimizer_path = os.path.join(optimizer_path)
    return [f for f in os.listdir(optimizer_path) if os.path.isdir(os.path.join(optimizer_path, f)) and 'num_epochs' in f]


def _check_if_metric_is_available(optimizer_path, metric):
    settings = _read_all_settings_folders(optimizer_path)
    sett = settings[0]
    path = os.path.join(optimizer_path, sett)
    run = [r for r in os.listdir(path) if '.json' in r][0]
    json_data = _load_json(path, run)
    if metric in json_data:
        return True
    else:
        return False


# TODO wherever it was used with warning, change from _check metric to _determine_metric
def _determine_available_metric(optimizer_path, metric):
    if _check_if_metric_is_available(optimizer_path, metric):
        return metric
    else:
        print('Metric {0:s} does not exist for testproblem {1:s}. We now use fallback metric \'test_losses\''.format(
            metric, os.path.split(os.path.split(optimizer_path)[0])[1]))
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


def _get_all_setting_analyzer(optimizer_path, metric = 'test_accuracies'):
    metric = _determine_available_metric(optimizer_path, metric)
    optimizer_path = os.path.join(optimizer_path)
    setting_folders = _read_all_settings_folders(optimizer_path)
    setting_analyzers = []
    for sett in setting_folders:
        sett_path = os.path.join(optimizer_path, sett)
        setting_analyzers.append(SettingAnalyzer(sett_path, metric))
    return setting_analyzers


def create_setting_analyzer_ranking(optimizer_path, mode = 'final', metric = 'test_accuracies'):
    metric = _determine_available_metric(optimizer_path, metric)
    setting_analyzers = _get_all_setting_analyzer(optimizer_path, metric)

    if 'accuracies' in metric:
        sgn = -1
    else:
        sgn = 1

    if mode == 'final':
        setting_analyzers_ordered = sorted(setting_analyzers, key=lambda idx: sgn * idx.final_value)
    elif mode == 'best':
        setting_analyzers_ordered = sorted(setting_analyzers, key=lambda idx: sgn * idx.best_value)
    elif mode == 'most':
        setting_analyzers_ordered = sorted(setting_analyzers, key=lambda idx: idx.n_runs, reverse=True)
    else:
        raise RuntimeError('Mode not implemented')

    return setting_analyzers_ordered


class SettingAnalyzer:
    """DeepOBS analyzer class for a setting (a hyperparameter setting).

    Args:
        path (str): Path to  the setting folder.
        metric (str): Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.

    Attributes:
        path
        metric (str): Metric to use for this test problem. If available this
            will be ``test_accuracies``, otherwise ``test_losses``.
        aggregate (dictionary): Contains the mean and std of the runs for the given metric.
        final_value
        best_value
    """
    def __init__(self, path, metric):
        """Initializes a new SettingAnalyzer instance.

        Args:
            path (str): String to the setting folder.
            metric (str): Metric to use for this test problem. If available this
                will be ``test_accuracies``, otherwise ``test_losses``.
        """

        self.path = path
        self.n_runs = self.__get_number_of_runs()
        self.aggregate = aggregate_runs(path)
        self.metric = metric
        self.final_value = self.__get_final_value()
        self.best_value = self.__get_best_value()

    def __get_number_of_runs(self):
        return len([run for run in os.listdir(self.path) if run.endswith(".json")])

    def __get_final_value(self):
        """Get final (mean) value of the metric used in this test problem.
        Returns:
            float: Final (mean) value of the test problem's metric.
        """
        return self.aggregate[self.metric]['mean'][-1]

    def __get_best_value(self):
        """Get best (mean) value of the metric used in this test problem.
        Returns:
            float: Best (mean) value of the test problem's metric.
        """
        if self.metric == 'test_losses' or self.metric == 'train_losses':
            return min(self.aggregate[self.metric]['mean'])
        elif self.metric == 'test_accuracies' or self.metric == 'train_accuracies':
            return max(self.aggregate[self.metric]['mean'])
        else:
            raise RuntimeError("Metric unknown")