.PHONY: help
.PHONY: black black-check flake8
.PHONY: install install-dev install-devtools install-test install-lint
.PHONY: test
.PHONY: conda-env

## taking ideas from https://github.com/foodkg/foodkg.github.io/blob/master/src/prep-scripts/Makefile

## pip env
env = env/$(if $(findstring Windows,$(OS)),Scripts,bin)/activate # crossplatform shenanigans
python = . $(env); python # source activate script before calling python
pip = $(python) -m pip # source activate script before calling pip

env: setup.py # create virtual environment - do not use $(python) before env exists!
	rm -rf env
	python -m venv env
	$(pip) install .


install: env
	$(pip) install .

install-dev: env
	$(pip) install -e .[doc,test,lint,git-hook]
	pre-commit install


.DEFAULT: help
help:
	@echo "test"
	@echo "        Run pytest on the project and report coverage"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "install"
	@echo "        Install deepobs and dependencies"
	@echo "install-dev"
	@echo "        Install all development tools"
	@echo "install-lint"
	@echo "        Install only the linter tools (included in install-dev)"
	@echo "install-test"
	@echo "        Install only the testing tools (included in install-dev)"
	@echo "conda-env"
	@echo "        Create conda environment 'deepobs' with dev setup"
###
# Test coverage
test:
	@pytest -vx --cov=deepobs .

###
# Linter and autoformatter

# Uses black.toml config instead of pyproject.toml to avoid pip issues. See
# - https://github.com/psf/black/issues/683
# - https://github.com/pypa/pip/pull/6370
# - https://pip.pypa.io/en/stable/reference/pip/#pep-517-and-518-support
black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

flake8:
	@flake8 .

###
# Installation


install-lint:
	@pip install -r requirements/lint.txt

install-test:
	@pip install -r requirements/test.txt

install-devtools:
	@echo "Install dev tools..."
	@pip install -r requirements-dev.txt
	@pip install -r requirements_doc.txt


###
# Conda environment
conda-env:
	@conda env create --file .conda_env.yml

