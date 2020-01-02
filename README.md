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

## TODO

- [ ] Add support for adding structural context via neuroimaging processed data (e.g. FreeSurfer)

# Installation Guide
EEGio is intended to be a lightweight wrapper for easily analyzing large batches of patients with EEG data. eegio relies on the following libraries to work:

    numpy
    scipy
    scikit-learn
    pandas
    mne
    mne-bids
    pybids
    seaborn
    matplotlib
    pyedflib (deprecated)
    xlrd (deprecated)
    
See [INSTALLATION GUIDE](./docs/INSTALLATION.md)

## Intended Users / Usage

Epilepsy researchers dealing with EEG data compliant with BIDS and MNE formats. Anyone with human patient EEG data.  
See example and docs for info on how to format this.

* https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md
* https://github.com/bids-standard/bids-starter-kit/wiki/The-BIDS-folder-hierarchy
    
### Setting Up the BIDS Directory and Reading EEG Data From EDF/FiF
These are just lightweight wrappers of MNE/pyedflib reading to load in EDF/FiF data
easily, so that raw EEG ts are readily accessible in Python friendly format. We provide
an example that was built off of the examples in MNE-BIDS.
See [example](./examples/read_eeg_from_bids.py).
    
For more info, see tutorials and documentation.

# Contributing
We welcome contributions from anyone. Please view our [contribution guidelines](./docs/CONTRIBUTING.md). Our [issues](https://github.com/adam2392/eegio/issues) page is a great place for suggestions!
If you have an idea for an improvement not listed there, please [make an issue](https://github.com/adam2392/eegio/issues/new) first so you can discuss with the developers. 
For information on setting up testing, see [testing guide](./docs/TESTING_SETUP.md).
    
# License

This project is covered under the **GNU GPL License**.
