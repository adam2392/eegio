"""
===================================
01. Convert EEG data to BIDS format
===================================

In this example, we use MNE-BIDS to create a BIDS-compatible directory of EEG
data. We copy some explicit example from MNE-BIDS and modify small details.

Specifically, we will follow these steps:

1. Download repository, and use the data in example directory:
    data/
        bids_layout/
            raw/

2. Load the source raw data, extract information, preprocess certain things
and save in a new BIDS directory

3. Check the result and compare it with the standard

"""

# Authors: Adam Li <adam2392@gmail.com>

import os

from bids_validator import BIDSValidator
from mne_bids import read_raw_bids, make_bids_folders, make_bids_basename
from mne_bids.utils import _parse_bids_filename, _handle_kind
from mne_bids.utils import print_dir_tree

from eegio.base.utils.bids_helper import BidsConverter
from eegio.loaders.bids.bids_run import BidsRun

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
RUN_IEEG = True  # either run scalp, or iEEG
line_freq = 60  # user should set the line frequency, since MNE-BIDS defaults to 50 Hz
test_subjectid = "0001"
test_sessionid = "seizure"
test_runid = "01"
if RUN_IEEG:
    edf_fpaths = [os.path.join(bids_root, "raw", "ieeg_ecog_test.edf")]
    modality = "ecog"
else:
    edf_fpath1 = os.path.join(bids_root, "raw", "scalp_test.edf")
    edf_fpath2 = os.path.join(bids_root, "raw", "scalp_test_2.edf")
    edf_fpaths = [edf_fpath1, edf_fpath2]
    modality = "eeg"
###############################################################################
# Let's see whether the data has been downloaded using a quick visualization
# of the directory tree.

data_dir = os.path.join(bids_root, "raw")
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
for edf_fpath in edf_fpaths:
    # create the BIDS directory structure
    if not os.path.exists(bids_root):
        print("Making bids root directory.")
        make_bids_folders(
            output_path=bids_root,
            session=test_sessionid,
            subject=test_subjectid,
            kind=modality,
        )

    """ Write data file into BIDS format """
    # add a bids run
    bids_basename = make_bids_basename(
        subject=test_subjectid,
        session=test_sessionid,
        run=test_runid,
        acquisition=modality,
    )
    print("Loading filepath: ", edf_fpath)
    print("Writing to bidsroot: ", bids_root)
    print("Bids basenmae; ", bids_basename)
    # call bidsbuilder pipeline
    bids_root = BidsConverter.convert_to_bids(
        edf_fpath=edf_fpath,
        bids_root=bids_root,
        bids_basename=bids_basename,
        acquisition=modality,
        line_freq=line_freq,
        overwrite=True,
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
bids_fname = bids_basename + f"_{kind}.fif"

print("Trying to read from: ", bids_fname)

# use MNE-BIDS function to read in the data
raw = read_raw_bids(bids_fname, bids_root)

print("Read successfully using MNE-BIDS")

# use BidsRun object, which just simply adds additional functionality
bidsrun = BidsRun(bids_root, bids_fname)
raw = bidsrun.load_data()

print(raw)

###############################################################################
# Step 4: Run BIDS-Validate
# ------------------------------------------------------------
# Now we have written our BIDS directory.
# save a fif copy and reload it
# TODO: figure out why this isn't returning valid, when it is valid.
params = _parse_bids_filename(bids_basename, True)
kind = _handle_kind(raw)
fif_data_path = make_bids_folders(
    subject=params["sub"],
    session=params["ses"],
    kind=kind,
    output_path=bids_root,
    overwrite=False,
    verbose=True,
)
rel_bids_root = "/bids_layout/sub-0001/ses-seizure/ieeg/"
is_valid = BIDSValidator().is_bids(os.path.join(rel_bids_root, bids_fname))

print("checked filepath: ", os.path.join(rel_bids_root, bids_fname))
print(is_valid)
