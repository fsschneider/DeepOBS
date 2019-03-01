# -*- coding: utf-8 -*-
"""Setup for the DeepOBS package"""

import os
import setuptools


def package_files(directory):
    """Get all json files.

    Args:
        directory (str): Path to parent dir.

    Returns:
        list: List of all json files.

    """
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.json'):
                paths.append(os.path.join('..', path, filename))
    return paths


BASELINE_FILES = package_files('deepobs/baselines/')

setuptools.setup(
    name='deepobs',
    version='1.1.0',
    description='Deep Learning Optimizer Benchmark Suite',
    author='Frank Schneider, Lukas Balles and Philipp Hennig,'
    'University of Tuebingen, Methods of Machine Learning',
    author_email='frank.schneider@tue.mpg.de',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'argparse', 'numpy', 'pandas', 'matplotlib', 'matplotlib2tikz',
        'seaborn'
    ],
    scripts=[
        'deepobs/scripts/deepobs_prepare_data.sh',
        'deepobs/scripts/deepobs_plot_results.py',
        'deepobs/scripts/deepobs_estimate_runtime.py'
    ],
    package_data={'': BASELINE_FILES},
    zip_safe=False)
