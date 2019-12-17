import os

from bids_validator import BIDSValidator
from mne_bids import (
    read_raw_bids,
    make_bids_folders,
    make_bids_basename,
    make_dataset_description,
)
from mne_bids.utils import _parse_bids_filename
from mne_bids.utils import print_dir_tree

from eegio.base.utils.bids_helper import BidsConverter

def get_all_subjectids(datadir):
    allpats = [p for p in os.listdir(datadir) if os.path.isdir(os.path.join(datadir,p))]
    return allpats

###############################################################################
# Step 1: Prepare the data
# -------------------------

DATADIR = "/Users/adam2392/Downloads/tngpipeline/"
bids_root = os.path.join(DATADIR)
centers = [
    "ummc",
    # "nih",
    # "jhu",
    # "cleveland",
    # "clevelandnl",
]
RUN_IEEG = False  # either run scalp, or iEEG
line_freq = 60  # user should set the line frequency, since MNE-BIDS defaults to 50 Hz
sessionid = "seizure"
task = "monitor"
authors = ["Adam Li"]
modality = "ecog"

edf_fpaths = []
subject_ids = []
for i, center in enumerate(centers):
    centerdir = os.path.join(bids_root, "sourcedata", center)

    allpats = get_all_subjectids(centerdir)

    for pat in allpats:
        edf_fpaths.append([os.path.join(centerdir, pat, "seeg", "edf", f)
                      for f in os.listdir(os.path.join(centerdir, pat, "seeg", "edf"))
                      if f.endswith(".edf")])
        subject_ids.append(pat)

    run_id = i

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
for subject_id, edf_fpaths in zip(subject_ids, edf_fpaths):
    for i, edf_fpath in enumerate(edf_fpaths):
        """ Write data file into BIDS format """
        test_runid = i

        # add a bids run
        bids_basename = make_bids_basename(
            subject=subject_id,
            session=sessionid,
            task=task,
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
            line_freq=line_freq,
            overwrite=True,
        )

# currently write_raw_bids overwrites make_dataset_description
# TODO: put this on the top when PR gets merged.
make_dataset_description(
    os.path.join(bids_root), name="epilepsy_ieeg", authors=authors
)

###############################################################################
# What does our fresh BIDS directory look like?
# print_dir_tree(bids_root)

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
# bidsrun = BidsRun(bids_root, bids_fname)
# raw = bidsrun.load_data()

print(raw)

###############################################################################
# Step 4: Run BIDS-Validate
# ------------------------------------------------------------
# Now we have written our BIDS directory.
# save a fif copy and reload it
# TODO: re-check when pybids is updated.
# currently, use the https://bids-standard.github.io/bids-validator/ and see that it is verified
params = _parse_bids_filename(bids_basename, True)
print(raw.info)
fif_data_path = make_bids_folders(
    subject=params["sub"],
    session=params["ses"],
    kind=kind,
    output_path=bids_root,
    overwrite=False,
    verbose=True,
)
rel_bids_root = f"/sub-0001/ses-seizure/{kind}/"
path = os.path.join(rel_bids_root, bids_fname)
is_valid = BIDSValidator().is_bids(path)

print(BIDSValidator().is_top_level(path))
print(BIDSValidator().is_associated_data(path))
print(BIDSValidator().is_session_level(path))
print(BIDSValidator().is_subject_level(path))
print(BIDSValidator().is_phenotypic(path))
print(BIDSValidator().is_file(path))

print("checked filepath: ", os.path.join(rel_bids_root, bids_fname))
print(is_valid)
