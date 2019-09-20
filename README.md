# EEG IO
[![Build Status](https://travis-ci.com/adam2392/eegio.svg?token=6sshyCajdyLy6EhT8YAq&branch=master)](https://travis-ci.com/adam2392/eegio)
[![Coverage Status](./coverage.svg)](./coverage.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![GitHub](https://img.shields.io/github/license/adam2392/eegio)
![PyPI](https://img.shields.io/pypi/v/eegio)
![GitHub last commit](https://img.shields.io/github/last-commit/adam2392/eegio)
<a href="https://codeclimate.com/github/adam2392/eegio/maintainability"><img src="https://api.codeclimate.com/v1/badges/2c7d5910e89350b967c8/maintainability" /></a>
![GitHub repo size](https://img.shields.io/github/repo-size/adam2392/eegio)

For an easy-to-use API interfacing with EEG data in EDF, or FIF format.

This module stores the code for IO of EEG data for human patients, and pipelining code to convert clinical center data (i.e. time series eeg, neuroimaging, clinical metadata) into a developer-friendly dataset that is also invertible and debug-friendly.

## Dev Process - TODO

- [ ] Add documentation
- [ ] Create fully-functional tests

## Intended Users / Usage

EZTrack team. Epilepsy researchers dealing with EEG data. Anyone with human patient EEG data. 

The main data workflow is to maintain a running Excel sheet with variables related to your patient population. 
See example and docs for info on how to format this.

User can run:

    eegio.format.format_clinical_sheet(excelfilepath)

User can run preprocessing on .edf files to create .fif + .json files:

    eegio.format.format_eegdata(edffilepath=edffilepath,
                                outputfilepath)
                                
User can then load datasets, or patient (i.e. grouped datasets) data:

    datasetobj = eegio.load.load_dataset(datasetfilepath)
    patientobj = eegio.load.load_patients(patientdir)
    print(datasetobj)
    print(patientobj)
    
### Reading Clinical Data From Excel
Most clinical data is stored in excel format. And changing between these two is ideal. We provide examples of
clinical data in table formats for multiple patients, single patient, or single snapshot. These
can be stored in a single Excel file if necessary. It just light-weight wraps Pandas, but
will provide additional functionality in terms of:

1. expanding channels using regex
2. preprocessing column names
3. caching/storing data in a Python object


    fpath = "./data/clinical_examples/test_clinicaldata.csv"
    # instantiate datasheet loader
    clinloader = DataSheet()

    # load csv file
    csv_df = clinloader.load(fpath=clinical_csv_fpath)

    # clean up the column names
    csv_df = clinloader.clean_columns(csv_df)
    
    # show input data
    display(csv_df.head())
    
### Reading EEG Data From EDF/FiF
These are just lightweight wrappers of MNE/pyedflib reading to load in EDF/FiF data
easily, so that raw EEG ts are readily accessible in Python friendly format.    

    edf_fpath = "./data/scalp_test.edf"
    read_kwargs = {
        "fname": edf_fpath,
        "backend": "mne",
        "montage": None,
        "eog": None,
        "misc": None,
    }
    loader = Loader(edf_fpath, metadata={})
    raw_mne, annotations = loader.read_edf(**read_kwargs)
    info = raw_mne.info
    chlabels = raw_mne.ch_names
    n_times = raw_mne.n_times

# Installation Guide
EEGio is intended to be a lightweight wrapper for easily analyzing large batches of patients with EEG data. eegio relies on the following libraries to work:

    numpy
    scipy
    scikit-learn
    seaborn
    pandas
    mne
    pyedflib
    
Setup environment via Conda:


    conda create -n eegio
    conda activate eegio
    conda config --add channels conda-forge
    conda install numpy pandas mne
    
## Install from Github
To install, run this command in your repo:

    pip install -e git+https://github.com/adam2392/eegio#egg=eegio


## Submodules
1. base/
Stores configuration files for the EDP and HARDCODED settings. Stores utility and helper functions.

2. format/
Stores class objects for how the EDP should preformat any incoming raw data. This defines the interface for the raw time series eeg, neuroimaging, and clinical metadata.

3. loaders/
This defines code that links together different parts of the eegio API to allow easy data pipelines to be setup.

4. writers/
This defines writers of data

5. dataset_test/
A dataset tester that is hardcoded to check quality of certain labeling and annotations.

# Documentation

    sphinx-quickstart
    

# Testing

    black eegio/*
    black tests/*
    pylint ./eegio/
    pytest --cov-config=.coveragerc --cov=./eegio/ tests/
    coverage-badge -f -o coverage.svg

Tests is organized into two directories right now: 
1. eegio/: consists of all unit tests related to various parts of eegio.
2. api/: consists of various integration tests tha test when eegio should work.

# License

This project is covered under the **GNU GPL License**.