# -*- coding: utf-8 -*-
"""Setup for the DeepOBS package"""

import setuptools
from deepobs import __version__

install_requires_list = [
        'argparse', 'numpy', 'pandas', 'matplotlib', 'matplotlib2tikz',
        'seaborn', 'bayesian-optimization'
    ]


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(
    name='deepobs',
    version=__version__,
    description='Deep Learning Optimizer Benchmark Suite',
    long_description=readme(),
    author='Frank Schneider, Lukas Balles and Philipp Hennig,',
    author_email='frank.schneider@tue.mpg.de',
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=install_requires_list,
    scripts=[
        'deepobs/scripts/deepobs_prepare_data.sh',
        'deepobs/scripts/deepobs_get_baselines.sh',
        'deepobs/scripts/deepobs_plot_results.py'
    ],
    zip_safe=False)
