# EEG IO
[![Build Status](https://travis-ci.com/adam2392/eegio.svg?token=6sshyCajdyLy6EhT8YAq&branch=master)](https://travis-ci.com/adam2392/eegio)
[![Coverage Status](./coverage.svg)](./coverage.svg)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
![GitHub](https://img.shields.io/github/license/adam2392/eegio)
![PyPI](https://img.shields.io/pypi/v/eegio)
![GitHub last commit](https://img.shields.io/github/last-commit/adam2392/eegio)
<a href="https://codeclimate.com/github/adam2392/eegio/maintainability"><img src="https://api.codeclimate.com/v1/badges/2c7d5910e89350b967c8/maintainability" /></a>
![GitHub repo size](https://img.shields.io/github/repo-size/adam2392/eegio)

For an easy-to-use API interfacing with EEG data in EDF, or FIF format.

This module stores the code for IO of EEG data for human patients, and pipelining code to convert clinical center data (i.e. time series eeg, neuroimaging, clinical metadata) into a developer-friendly dataset that is also invertible and debug-friendly.

## Dev Process - TODO

- [ ] Add documentation badge: 
- [ ] Create fully-functional tests
- [ ] Support for edf, fif, mat files
- [ ] Preprocessed pipeline first
- [ ] Separate excel reading into metadata

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

    pip install git+https://github.com/adam2392/eegio#egg=eegio


## Submodules
1. base/
Stores configuration files for the EDP and HARDCODED settings.

2. format/
Stores class objects for how the EDP should preformat any incoming raw data.

This defines the interface for the raw time series eeg, neuroimaging, and clinical metadata.

3. loaders/
This defines code that links together different parts of the EDP API to allow easy data pipelines to be setup.

I define preformat pipeline, neuroimaging pipeline, TBD.

4. util/
Stores utility and helper functions.

# License

This project is covered under the **GNU GPL License**.