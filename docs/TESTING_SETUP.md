# Guide to setup Testing Environment

# Testing Scipts

    black eegio/*
    black tests/*
    black --check eegio/
    pytest --cov-config=.coveragerc --cov=./eegio/ tests/ 
    coverage-badge -f -o coverage.svg

Tests is organized into two directories right now: 
1. eegio/: consists of all unit tests related to various parts of eegio.
2. api/: consists of various integration tests tha test when eegio should work.

# Updating Packages

    conda update --all
    pip freeze > requirements.txt
    conda env export > environment.yaml

# Documentation 
# TODO: setup guides for running autodoc

    sphinx-quickstart