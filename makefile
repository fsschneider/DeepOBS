# MAKEFILE

.PHONY: help
.PHONY: clean-all clean-pyc clean-test

.PHONY: lint, flake8, black, pydocstyle, darglint, isort

.PHONY: build-docs

.PHONY: conda_env

help:
	@echo "**clean-all"
	@echo "  Removes all unnecessary files."
	@echo " *clean-pyc"
	@echo "  Removes all Python file artifacts, e.g. *.pyc files."
	@echo " *clean-test"
	@echo "  Removes all Python testing artifcats, e.g. .pytest_cache."
	@echo "**lint"
	@echo "  Checks the whole code for formatting and linting errors via flake8, black, pydocstyle, darglint and isort."
	@echo " *black"
	@echo "  Run Black formatter to check whether it would change files."
	@echo " *flake8"
	@echo "  Run Flake8."
	@echo " *pydocstyle"
	@echo "  Run Pydocstyle."
	@echo " *darglint"
	@echo "  Run darglint."
	@echo " *isort"
	@echo "  Run isort."
	@echo "**build-docs"
	@echo "  Build the docs."
	@echo "**conda_env"
	@echo "  Create the conda environment for the project."

### CLEAN ###
clean-all: clean-pyc clean-test

# Removes all pyc and __pycach__ files
clean-pyc:
	@find . -name '*.pyc' -delete
	@find . -name '*.pyo' -delete
	@find . -name '*~' -delete
	@find . -type d -name "__pycache__" -delete

# Removes the pytest_cache and benchmark directories
clean-test:
	@rm -fr .pytest_cache/
	@rm -fr .benchmarks/

### LINTING ###
lint: black flake8 pydocstyle

black:
	@black . --check

flake8:
	@flake8

pydocstyle:
	@pydocstyle --count .

darglint:
	@darglint --verbosity 2

isort:
	@isort . --check

### DOCS ###
build-docs:
	@cd docs && make clean && make html

### CONDA ###
conda-env:
	@conda env create --file .conda_env.yml