""" Define fixtures available for eegio tests. """
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
def bids_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("bids_data")
    return str(fn)


@pytest.fixture(scope="session")
def test_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("bids_data")
    return str(fn)


@pytest.fixture(scope="class")
def bidspatient(tmp_path_factory):
    """
    Creating a pytest fixture for a fake bidsPatient data object.

    :return:
    """
    test_subjectid = "0001"
    test_fpath = os.path.join(os.getcwd(), "data/bids_layout/")
    # test_fpath_path = tmp_path_factory.mktemp('bids_data')
    # test_fpath = str(test_fpath_path)
    bidspat = BidsPatient(
        subject_id=test_subjectid,
        session_id="seizure",
        datadir=test_fpath,
        managing_user="test",
        modality="eeg",
    )
    return bidspat


@pytest.fixture(scope="class")
def bidsrun_scalp(tmp_path_factory, edf_fpath):
    """
    Creating a pytest fixture for a fake bidsRun data object.
    This dataset should already work for the .fif dataset!

    EDF will not match channel names.

    Returns
    -------

    """
    test_subjectid = "0001"
    session_id = "seizure"
    kind = "eeg"
    run_id = "01"
    bids_fname = make_bids_basename(
        subject=test_subjectid,
        session=session_id,
        acquisition=kind,
        run=run_id,
        suffix=kind + ".fif",
    )
    test_bids_root = os.path.join(os.getcwd(), "./data/bids_layout/")
    run = BidsRun(bids_root=test_bids_root, bids_fname=bids_fname)
    return run


@pytest.fixture(scope="function")
def contacts():
    """
    Creating a pytest fixture for a fake Contacts class init.

    :return:
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
def clinical_fpath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/clinical_examples/")
    filepath = os.path.join(testdatadir, "test_clinicaldata.csv")
    return filepath


@pytest.fixture(scope="class")
def edf_fpath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/bids_layout/raw/")
    filepath = os.path.join(testdatadir, "scalp_test.edf")
    return filepath


@pytest.fixture(scope="class")
def fif_fpath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/bids_layout/raw/")
    filepath = os.path.join(testdatadir, "scalp_test_raw.fif")
    return filepath


@pytest.fixture(scope="class")
def test_arr():
    test_arr = np.random.random(size=(50, 2500))

    samplepoints = compute_samplepoints(250, 125, test_arr.shape[1])

    # split into test windows
    test_list = []
    for i, (samplestart, sampleend) in enumerate(samplepoints):
        test_list.append(test_arr[:, samplestart:sampleend])

    return test_list


@pytest.fixture(scope="class")
def result_fpath():
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/result_examples")
    result_fpath = os.path.join(testdatadir, "test_fragmodel.json")
    result_npzfpath = os.path.join(testdatadir, "test_fragmodel.npz")
    return result_fpath, result_npzfpath
