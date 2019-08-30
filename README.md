# EEG IO
[![Build Status](https://travis-ci.com/adam2392/eegio.svg?token=6sshyCajdyLy6EhT8YAq&branch=master)](https://travis-ci.com/adam2392/eegio)
[![Coverage Status](./coverage.svg)](./coverage.svg)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

For an easy-to-use API interfacing with EEG data in EDF, or FIF format.

This module stores the code for IO of EEG data for human patients, and pipelining code to convert clinical center data (i.e. time series eeg, neuroimaging, clinical metadata) into a developer-friendly dataset that is also invertible and debug-friendly.

## Intended Users

EZTrack team. Epilepsy researchers dealing with EEG data. Anyone with human patient EEG data.

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
