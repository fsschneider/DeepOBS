# -*- coding: utf-8 -*-
"""Setup for the DeepOBS package."""

import setuptools

install_requires_list = [
    "tensorflow~=2.5",
    "tensorflow-addons~=0.13.0",
    "argparse",
    "bayesian-optimization",
    "matplotlib",
    "numpy",
    "pandas",
    "seaborn",
    "tikzplotlib",
]


def readme():
    """Read the Readme file.

    Returns:
        str: Content of the README.md file
    """
    # for some reason autodetects "charmap" encoding on windows -> explicit encoding
    with open("README.md", mode="r", encoding="utf-8") as f:
        return f.read()


version_dict = {}
exec(open("deepobs/version.py").read(), version_dict)

setuptools.setup(
    name="deepobs",
    version=version_dict["__version__"],
    description="Deep Learning Optimizer Benchmark Suite",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Frank Schneider, Aaron Bahde, Lukas Balles, and Philipp Hennig",
    author_email="frank.schneider@tue.mpg.de",
    url="https://github.com/fsschneider/deepobs",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=install_requires_list,
    scripts=[
        "deepobs/scripts/deepobs_prepare_data.sh",
        "deepobs/scripts/deepobs_get_baselines.sh",
        "deepobs/scripts/deepobs_plot_results.py",
    ],
    zip_safe=False,
)
