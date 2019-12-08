# EEG IO
[![Build Status](https://travis-ci.com/adam2392/eegio.svg?token=6sshyCajdyLy6EhT8YAq&branch=master)](https://travis-ci.com/adam2392/eegio)
[![Coverage Status](./coverage.svg)](./coverage.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![GitHub](https://img.shields.io/github/license/adam2392/eegio)
![PyPI](https://img.shields.io/pypi/v/eegio)
![GitHub last commit](https://img.shields.io/github/last-commit/adam2392/eegio)
<a href="https://codeclimate.com/github/adam2392/eegio/maintainability"><img src="https://api.codeclimate.com/v1/badges/2c7d5910e89350b967c8/maintainability" /></a>
![GitHub repo size](https://img.shields.io/github/repo-size/adam2392/eegio)

For an easy-to-use API interfacing with EEG data in EDF, or FIF format in the BIDS-EEG layout.

This module stores the code for IO of EEG data for human patients, and pipelining code to convert clinical center data (i.e. time series eeg, clinical metadata) into a developer-friendly dataset that is also invertible and debug-friendly.

## Dev Process - TODO

- [ ] Add support for adding structural context via neuroimaging processed data (e.g. FreeSurfer)

# Installation Guide
EEGio is intended to be a lightweight wrapper for easily analyzing large batches of patients with EEG data. eegio relies on the following libraries to work:

    numpy
    scipy
    scikit-learn
    seaborn
    pandas
    mne
    mne-bids
    pybids
    pyedflib (deprecated)
    xlrd (deprecated)
    
Setup virtual environment via Conda inside your Unix-friendly terminal (aka Mac, or Linux) is recommended (see https://docs.anaconda.com/anaconda/install/):


    conda create -n eegio # creates conda env
    conda activate eegio  # activates the environment
    conda config --add channels conda-forge # add extra channels necessary
    conda install numpy pandas mne scikit-learn scipy seaborn matplotlib pyedflib xlrd
    
## Install from Github (Mainly for developing)
To install, run this command inside your virtual environment:

    pip install -e git+https://github.com/adam2392/eegio#egg=eegio


## Intended Users / Usage

Epilepsy researchers dealing with EEG data compliant with BIDS and MNE formats. Anyone with human patient EEG data.  
See example and docs for info on how to format this.

https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md
    
### Reading EEG Data From EDF/FiF
These are just lightweight wrappers of MNE/pyedflib reading to load in EDF/FiF data
easily, so that raw EEG ts are readily accessible in Python friendly format.    

    TODO: example read using bidsrun, bidspatient
    
### Setting Up the BIDS Directory
See tutorials and documentation.

## Submodules
1. base/
Stores configuration files for the settings. Stores utility and helper functions. In addition, defines dataset objects
that we use in this module: mainly Contacts, EEGTimeSeries, and Result

2. loaders/
This defines code that links together different parts of the eegio API to allow easy data pipelines to be setup.

3. writers/
This defines writers of data to save into hdf, npy, mat files.

# Contributing
We welcome contributions from anyone. Our [issues](https://github.com/adam2392/eegio/issues) page is full of places we could use help! 
If you have an idea for an improvement not listed there, please [make an issue](https://github.com/adam2392/eegio/issues/new) first so you can discuss with the developers. 

# Documentation

    sphinx-quickstart
    

# Testing

    black eegio/*
    black tests/*
    pylint ./eegio/
    pre-commit run black --all-files
    pytest --cov-config=.coveragerc --cov=./eegio/ tests/ 
    coverage-badge -f -o coverage.svg

Tests is organized into two directories right now: 
1. eegio/: consists of all unit tests related to various parts of eegio.
2. api/: consists of various integration tests tha test when eegio should work.

# Updating Packages

    conda update --all
    pip freeze > requirements.txt
    conda env export > environment.yaml

# License

This project is covered under the **GNU GPL License**.