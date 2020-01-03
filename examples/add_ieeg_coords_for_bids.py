"""
=====================================================
02. Convert iEEG data with coordinates to BIDS format
=====================================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of EEG
data. We copy some explicit example from MNE-BIDS and modify small details.
"""

# Authors: Adam Li <adam2392@gmail.com>

import os
import numpy as np

from bids_validator import BIDSValidator
import mne
from mne_bids import (
    read_raw_bids,
    make_bids_folders,
    make_bids_basename,
    make_dataset_description,
)
from mne_bids.utils import _parse_bids_filename
from mne_bids.utils import print_dir_tree

from eegio.loaders import BidsRun
from eegio.base.utils.bids_helper import BidsConverter

###############################################################################
# Step 1: Prepare the data
# -------------------------
#
# First, we need some data to work with. We will use some sample simulated scalp and
# iEEG data. For each subject, there are "seizure" events. For the present example, we will
# show how to format the data for two modalities to comply with the Brain Imaging Data Structure
# (`BIDS <http://bids.neuroimaging.io/>`_).
#
# The data are in the `European Data Format <https://www.edfplus.info/>`_
# '.edf', which is good for us because next to the BrainVision format, EDF is
# one of the recommended file formats for EEG BIDS. However, apart from the
# data format, we need to build a directory structure and supply meta data
# files to properly *bidsify* this data.
#
# Conveniently, there is already a data loading function available with
# MNE-Python:

DATADIR = os.getcwd()
bids_root = os.path.join(DATADIR, "../data/bids_layout/")
line_freq = 60  # user should set the line frequency, since MNE-BIDS defaults to 50 Hz
test_subjectid = "0001"
test_sessionid = "seizure"
test_task = "monitor"
authors = ["Adam Li", "Patrick Myers"]
edf_fpaths = [os.path.join(bids_root, "sourcedata", "ieeg_ecog_test.edf")]
modality = "ecog"

###############################################################################
# Let's see whether the data has been downloaded using a quick visualization
# of the directory tree.

data_dir = os.path.join(bids_root, "sourcedata")
print_dir_tree(data_dir)

###############################################################################
# Step 2: Formatting as BIDS
# --------------------------
#
# Let's start by formatting a single subject. We are reading the data using
# MNE-Python's io module and the :func:`read_raw_edf` function. Note that we
# must use `preload=False`, the default in MNE-Python. It prevents the data
# from being loaded and modified when converting to BIDS.
#
# Note that kind and acquisition currently almost stand for the same thing.
# Please read the BIDS docs to get acquainted.

# create the BIDS directory structure
if not os.path.exists(bids_root):
    print("Making bids root directory.")
    make_bids_folders(
        output_path=bids_root,
        session=test_sessionid,
        subject=test_subjectid,
        kind=modality,
    )

coords_fpath = os.path.join(bids_root, "sourcedata", "ieeg_ecog_coords.txt")

for i, edf_fpath in enumerate(edf_fpaths):
    """ Write data file into BIDS format """
    test_runid = i

    # add a bids run
    bids_basename = make_bids_basename(
        subject=test_subjectid,
        session=test_sessionid,
        task=test_task,
        run=test_runid,
        acquisition=modality,
    )

    print("Loading filepath: ", edf_fpath)
    print("Writing to bidsroot: ", bids_root)
    print("Bids basenmae; ", bids_basename)
    # call bidsbuilder pipeline
    bids_deriv_root = BidsConverter.convert_to_bids(
        edf_fpath=edf_fpath,
        bids_root=bids_root,
        bids_basename=bids_basename,
        coords_fpath=coords_fpath,
        line_freq=line_freq,
        overwrite=True,
    )

# currently write_raw_bids overwrites make_dataset_description
# TODO: put this on the top when PR gets merged.
make_dataset_description(
    os.path.join(bids_root), name="test_bids_dataset", authors=authors
)

###############################################################################
# What does our fresh BIDS directory look like?
print_dir_tree(bids_root)

###############################################################################
# Step 3: Check and compare and read in the data
# ------------------------------------------------------------
# Now we have written our BIDS directory.

if modality in ["ecog", "seeg"]:
    kind = "ieeg"
elif modality == "eeg":
    kind = "eeg"
bids_fname = bids_basename + f"_{kind}.edf"

print("Trying to read from: ", bids_fname)

# use MNE-BIDS function to read in the data
raw = read_raw_bids(bids_fname, bids_root)

print("Read successfully using MNE-BIDS")

# use BidsRun object, which just simply adds additional functionality
# bidsrun = BidsRun(tmp_bids_root, bids_fname)
# raw = bidsrun.load_data()

print(raw)

###############################################################################
# Step 4: Run BIDS-Validate
# ------------------------------------------------------------
# Now we have written our BIDS directory.
# save a fif copy and reload it
# TODO: re-check when pybids is updated.
# currently, use the https://bids-standard.github.io/bids-validator/ and see that it is verified
# params = _parse_bids_filename(bids_basename, True)
# print(raw.info)
# fif_data_path = make_bids_folders(
#     subject=params["sub"],
#     session=params["ses"],
#     kind=kind,
#     output_path=tmp_bids_root,
#     overwrite=False,
#     verbose=True,
# )
# rel_bids_root = f"/sub-0001/ses-seizure/{kind}/"
# path = os.path.join(rel_bids_root, bids_fname)
# is_valid = BIDSValidator().is_bids(path)
#
# print(BIDSValidator().is_top_level(path))
# print(BIDSValidator().is_associated_data(path))
# print(BIDSValidator().is_session_level(path))
# print(BIDSValidator().is_subject_level(path))
# print(BIDSValidator().is_phenotypic(path))
# print(BIDSValidator().is_file(path))
#
# print("checked filepath: ", os.path.join(rel_bids_root, bids_fname))
# print(is_valid)
