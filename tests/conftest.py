"""Define fixtures available for eegio tests."""
import os
import os.path

import numpy as np
import pytest
from mne_bids import make_bids_basename

from eegio.base.objects.electrodes.elecs import Contacts
from eegio.base.utils.data_structures_utils import compute_samplepoints
from eegio.loaders import BidsPatient
from eegio.loaders import BidsRun


@pytest.fixture(scope="session")
def bids_root(tmp_path_factory):
    """
    Fixture of a temporary path that can be used as a bids_directory.

    Should be used for:
    - testing bids setup
    - testing bids reading after setup
    - testing bids writing after setup

    Parameters
    ----------
    tmp_path_factory :

    Returns
    -------
    bids_root : (str) the temporary path to a bids_root
    """
    bids_root = tmp_path_factory.mktemp("bids_data")
    return str(bids_root)


# @pytest.fixture(scope="class")
# def bidspatient(tmp_path_factory):
#     """
#     Creating a pytest fixture for a fake bidsPatient data object.
#
#     :return:
#     """
#     test_subjectid = "0001"
#     test_fpath = os.path.join(os.getcwd(), "data/bids_layout/")
#     # test_fpath_path = tmp_path_factory.mktemp('bids_data')
#     # test_fpath = str(test_fpath_path)
#     bidspat = BidsPatient(
#         subject_id=test_subjectid,
#         session_id="seizure",
#         datadir=test_fpath,
#         managing_user="test",
#         modality="eeg",
#     )
#     return bidspat


@pytest.fixture(scope="class")
def bidsrun_scalp(edf_fpath):
    """
    Fixture of a fake bidsRun data object.

    Should be used for:
    - testing bids-run reading
    - testing bids-run writing

    Expected to work for .EDF and .FIF files.

    Parameters
    ----------
    edf_fpath : (str) the local test edf filepath.

    Returns
    -------
    run : (BidsRun object)
    """
    main_eeg()
    main_ieeg()
    
    test_subjectid = "0001"
    session_id = "seizure"
    kind = "eeg"
    run_id = "01"
    task = "monitor"
    bids_fname = make_bids_basename(
        subject=test_subjectid,
        session=session_id,
        acquisition=kind,
        task=task,
        run=run_id,
        suffix=kind + ".fif",
    )
    test_bids_root = os.path.join(os.getcwd(), "./data/bids_layout/derivatives")
    run = BidsRun(bids_root=test_bids_root, bids_fname=bids_fname)
    return run


@pytest.fixture(scope="function")
def contacts():
    """
    Fixture for a fake Contacts class.

    Returns
    -------
    contacts : (Contacts object)
    """
    contactlist = np.hstack(
        (
            [f"A{i}" for i in range(16)],
            [f"L{i}" for i in range(16)],
            [f"B'{i}" for i in range(16)],
            [f"D'{i}" for i in range(16)],
            ["C'1", "C'2", "C'4", "C'8"],
            ["C1", "C2", "C3", "C4", "C5", "C6"],
        )
    )
    contacts = Contacts(contactlist)
    return contacts


@pytest.fixture(scope="class")
def edf_fpath():
    """
    Fixture of a test sample edf filepath.

    Returns
    -------
    filepath : (str)
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/bids_layout/sourcedata/")
    filepath = os.path.join(testdatadir, "scalp_test.edf")
    return filepath


@pytest.fixture(scope="class")
def fif_fpath():
    """
    Fixture of a test sample edf filepath.

    Returns
    -------
    filepath : (str)
    """
    main_eeg()
    main_ieeg()

    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/bids_layout/sourcedata/")
    filepath = os.path.join(testdatadir, "scalp_test_raw.fif")
    return filepath


@pytest.fixture(scope="class")
def test_arr():
    """
    Fixture of a test sample numpy array with saved results matrix style.

    Should test:
    - any operations on sequentially saved array data

    Returns
    -------
    test_list : (list) a List of array based windows.
    """
    test_arr = np.random.random(size=(50, 2500))

    samplepoints = compute_samplepoints(250, 125, test_arr.shape[1])

    # split into test windows
    test_list = []
    for i, (samplestart, sampleend) in enumerate(samplepoints):
        test_list.append(test_arr[:, samplestart:sampleend])

    return test_list


@pytest.fixture(scope="class")
def result_fpath():
    """
    Fixture of a test sample numpy array with saved results matrix style.

    Returns
    -------
    result_fpath : (str)
    result_npzfpath : (str)
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/result_examples")
    result_fpath = os.path.join(testdatadir, "test_fragmodel.json")
    result_npzfpath = os.path.join(testdatadir, "test_fragmodel.npz")
    return result_fpath, result_npzfpath

def main_ieeg():
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
                sourcedata/
                derivatives/
                sub-XXX/
                sub-XXY/
                ...

    2. Load the source raw data, extract information, preprocess certain things
    and save in a new BIDS directory

    3. Check the result and compare it with the standard

    """

    # Authors: Adam Li <adam2392@gmail.com>

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
    test_task = "monitor"
    authors = ["Adam Li", "Patrick Myers"]

    if RUN_IEEG:
        edf_fpaths = [os.path.join(bids_root, "sourcedata", "ieeg_ecog_test.edf")]
        modality = "ecog"
    else:
        edf_fpath1 = os.path.join(bids_root, "sourcedata", "scalp_test.edf")
        edf_fpath2 = os.path.join(bids_root, "sourcedata", "scalp_test_2.edf")
        edf_fpaths = [edf_fpath1, edf_fpath2]
        modality = "eeg"
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
            bids_root=bids_root,
            session=test_sessionid,
            subject=test_subjectid,
            kind=modality,
        )

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
        bids_root=bids_root,
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

def main_eeg():
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
                sourcedata/
                derivatives/
                sub-XXX/
                sub-XXY/
                ...

    2. Load the source raw data, extract information, preprocess certain things
    and save in a new BIDS directory

    3. Check the result and compare it with the standard

    """

    # Authors: Adam Li <adam2392@gmail.com>

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
    RUN_IEEG = False  # either run scalp, or iEEG
    line_freq = 60  # user should set the line frequency, since MNE-BIDS defaults to 50 Hz
    test_subjectid = "0001"
    test_sessionid = "seizure"
    test_task = "monitor"
    authors = ["Adam Li", "Patrick Myers"]

    if RUN_IEEG:
        edf_fpaths = [os.path.join(bids_root, "sourcedata", "ieeg_ecog_test.edf")]
        modality = "ecog"
    else:
        edf_fpath1 = os.path.join(bids_root, "sourcedata", "scalp_test.edf")
        edf_fpath2 = os.path.join(bids_root, "sourcedata", "scalp_test_2.edf")
        edf_fpaths = [edf_fpath1, edf_fpath2]
        modality = "eeg"
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
            bids_root=bids_root,
            session=test_sessionid,
            subject=test_subjectid,
            kind=modality,
        )

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
        bids_root=bids_root,
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
