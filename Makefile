

PYTHON ?= python
PYTESTS ?= pytest

all: clean inplace test

clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "*.DS_Store" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pychache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

inplace:
	$(PYTHON) setup.py install

test: inplace
	rm -f .coverage
	$(PYTESTS) ./

test-doc:
	$(PYTESTS) --doctest-modules --doctest-ignore-import-errors

test-coverage:
	rm -rf coverage .coverage
	$(PYTESTS) --cov=./ --cov-report html:coverage

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

build-doc:
	cd doc; make clean
	cd doc; make html

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle

pycodestyle:
	@echo "Running pycodestyle"
	@pycodestyle

