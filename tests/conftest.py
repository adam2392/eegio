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
