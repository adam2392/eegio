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

This module stores the code for IO of EEG data for human patients, and pipelining code to convert clinical center data (i.e. time series eeg, clinical metadata) into a developer-friendly dataset that is also invertible and debug-friendly.

## Dev Process - TODO

- [ ] Add documentation
- [ ] Create fully-functional tests
- [ ] Add support for adding structural context via neuroimaging processed data (e.g. FreeSurfer)

# Installation Guide
EEGio is intended to be a lightweight wrapper for easily analyzing large batches of patients with EEG data. eegio relies on the following libraries to work:

    numpy
    scipy
    scikit-learn
    seaborn
    pandas
    mne
    pyedflib
    xlrd
    
Setup virtual environment via Conda inside your Unix-friendly terminal (aka Mac, or Linux) is recommended (see https://docs.anaconda.com/anaconda/install/):


    conda create -n eegio # creates conda env
    conda activate eegio  # activates the environment
    conda config --add channels conda-forge # add extra channels necessary
    conda install numpy pandas mne scikit-learn scipy seaborn matplotlib pyedflib xlrd
    
## Install from Github (Mainly for developing)
To install, run this command inside your virtual environment:

    pip install -e git+https://github.com/adam2392/eegio#egg=eegio


## Intended Users / Usage

Epilepsy researchers dealing with EEG data. Anyone with human patient EEG data. The main data workflow is to maintain a running Excel sheet with variables related to your patient population. 
See example and docs for info on how to format this.

## Formatting
User can run:

    formatted_df = eegio.format_clinical_sheet(excelfilepath,
                                            cols_to_reg_expand=["bad_channels"])

User can run preprocessing on .edf files to create .fif + .json files:

    rawmne, metadata = eegio.format_eegdata(in_fpath=edffilepath,
                        out_fpath=outputfilepath,
                        json_fpath=jsonfpath)

## Loading
User can then load datasets, or patient (i.e. grouped datasets) data:
    
    # create a patient object that looks through hard coded directory
    patientobj = eegio.get_patients(patientdir)
    
    # actually load an EEG Time series from .fif/.edf,etc.
    datasetobj = eegio.load_file(datasetfilepath)
    
    # read in metadata related from an excel file
    clin_dict = eegio.load_clinicalmetadata(formatted_df)
    
    # attach clinical data to metadata for EEG TimeSeries
    datasetobj.update_metadata(**clin_dict)
    
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
    clinloader = DataSheetLoader()

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

## Submodules
1. base/
Stores configuration files for the settings. Stores utility and helper functions. In addition, defines dataset objects
that we use in this module: mainly Contacts, EEGTimeSeries, and Result

2. format/
Stores class objects for how you should preformat any incoming raw data. This defines the interface for the raw time series eeg, and clinical metadata.

3. loaders/
This defines code that links together different parts of the eegio API to allow easy data pipelines to be setup.

4. writers/
This defines writers of data to save into hdf, npy, mat files.

5. dataset_test/
A dataset tester that is hardcoded to check quality of certain labeling and annotations.

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