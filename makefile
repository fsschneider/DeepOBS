.PHONY: help
.PHONY: black black-check flake8
.PHONY: install install-tf install-torch install-dev install-doc install-test install-lint
.PHONY: test
.PHONY: conda-env

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
	@echo "install-tf"
	@echo "        Install deepobs, dependencies and tensorflow"
	@echo "install-torch"
	@echo "        Install deepobs, dependencies and pytorch"
	@echo "install-dev"
	@echo "        Install all development tools"
	@echo "install-doc"
	@echo "        Install only the documentation tools (included in install-dev)"
	@echo "install-lint"
	@echo "        Install only the linter tools (included in install-dev)"
	@echo "install-test"
	@echo "        Install only the testing tools (included in install-dev)"
	@echo "conda-env"
	@echo "        Create conda environment 'deepobs' with dev setup"

### 
# Installation
## taking ideas from https://github.com/foodkg/foodkg.github.io/blob/master/src/prep-scripts/Makefile

env = env/$(if $(findstring Windows,$(OS)),Scripts,bin)/activate # crossplatform shenanigans
python = . $(env); python # source activate script before calling python
pip = $(python) -m pip # source activate script before calling pip

env: # create virtual environment - do not use $(python) before env exists!
	rm -rf env
	python -m venv env

install: env setup.py 
	$(pip) install .

install-tf: env setup.py
	$(pip) install .[tf]

install-torch: env setup.py
	$(pip) install .[torch]

# dev includes [doc,test,lint] and "pre-commit"
install-dev: env setup.py
	$(pip) install -e .[dev]
	pre-commit install

install-doc: env setup.py
	$(pip) install -e .[doc]
install-test: env setup.py
	$(pip) install -e .[test]
install-lint: env setup.py
	$(pip) install -e .[lint]

###
# Test coverage
test: env install-lint
	@. $(env); pytest -vx --cov=deepobs tests

###
# Linter and autoformatter

black: env install-lint
	@. $(env); black .

black-check: env install-lint
	@. $(env); black . --check

flake8: env install-lint
	@. $(env); flake8 .

###
# Conda environment
conda-env:
	@conda env create --file .conda_env.yml

