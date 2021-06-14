# -*- coding: utf-8 -*-
"""Setup for the DeepOBS package."""

import setuptools

install_requires_list = [
    "argparse",
    "bayesian-optimization",
    "matplotlib",
    "numpy",
    "pandas",
    "seaborn",
    "tikzplotlib",
]

tensorflow_requires_list = [
    "tensorflow~=1",
]

pytorch_requires_list = ["torch"]

docs_requires_list = (
    [
        "sphinx~=1.8.1",
        "sphinx-rtd-theme~=0.4.2",
        "sphinx-argparse~=0.2.3",
    ]
    + tensorflow_requires_list
    + pytorch_requires_list
)

tests_require_list = (
    [
        "pytest",
        "pytest-cov",
        "coveralls",
    ]
    + tensorflow_requires_list
    + pytorch_requires_list
)

lint_require_list = [
    "flake8",
    "mccabe",
    "pycodestyle",
    "pyflakes",
    "pep8-naming",
    "flake8-bugbear",
    "flake8-comprehensions",
    "black",
]

dev_require_list = list(
    set(["pre-commit"] + docs_requires_list + tests_require_list + lint_require_list)
)


def readme():
    """Read the Readme file.

    Returns:
        str: Content of the README.md file
    """
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
    extras_require={
        "tf": tensorflow_requires_list,
        "tensorflow": tensorflow_requires_list,
        "torch": pytorch_requires_list,
        "pytorch": pytorch_requires_list,
        "doc": docs_requires_list,
        "test": tests_require_list,
        "lint": lint_require_list,
        "dev": dev_require_list,
    },
    scripts=[
        "deepobs/scripts/deepobs_prepare_data.sh",
        "deepobs/scripts/deepobs_get_baselines.sh",
        "deepobs/scripts/deepobs_plot_results.py",
    ],
    zip_safe=False,
)
