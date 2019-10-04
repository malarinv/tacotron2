# Suitable for python monorepo with packages in subdirectories(contains project.mk)
.PHONY : _forward Makefile common.mk
.DEFAULT_GOAL := help
MAKE_PATH := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

include ${MAKE_PATH}/common.mk

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 sia tests

test: ## run tests quickly with the default Python
	python setup.py test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source sia -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/sia.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ sia
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist pypi ## package and upload a release to local pypi server
	twine upload -u "" -p "" --repository-url http://localhost:8080 dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean pypi ## install the package to the active Python's site-packages
	python setup.py install

check:  ## lint project using pre-commit hooks installed for git
	pre-commit run --all-files

setup:
	python -m nltk.downloader stopwords
	rm -rf logs
	mkdir logs

install-reqs:
	pip install -r requirements.txt

install-dev-reqs:
	pip install -r requirements_dev.txt

install: uninstall pypi install-reqs setup ## installs the requirements and download components

uninstall:  ## uninstalls the pip dependencies
	pip uninstall -y -r requirements.txt

deploy:  ## deploys the services by starting supervisord
	supervisord

develop: uninstall pypi install-dev-reqs setup  ## installs the requirements and setup development hooks
	pre-commit install
