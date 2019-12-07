""" Define fixtures available for eegio tests. """
import os
import os.path

import numpy as np
import pytest


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
    from eegio.loaders.bids import BidsPatient

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
    Creating a pytest fixture for a fake bidsRun data object
    Returns
    -------

    """
    from eegio.loaders.bids import BidsRun

    test_subjectid = "0001"
    session_id = "seizure"
    kind = "eeg"
    run_id = "01"
    test_fpath = os.path.join(os.getcwd(), "./data/bids_layout/")
    # test_fpath_path = tmp_path_factory.mktemp('bids_data')
    # test_fpath = str(test_fpath_path)
    # original_fileid = edf_fpath
    #
    # # Necessary within the temporary directory structure to first create the patient.
    # bidspat = BidsPatient(subject_val=test_subjectid,
    #                       session_id=session_id,
    #                       datadir=test_fpath,
    #                       kind=kind)
    # bidspat.add_scans([original_fileid])

    run = BidsRun(
        subject_id=test_subjectid,
        session_id=session_id,
        run_id=run_id,
        datadir=test_fpath,
        modality=kind,
    )
    return run


@pytest.fixture(scope="function")
def contacts():
    """
    Creating a pytest fixture for a fake Contacts class init.

    :return:
    """
    import numpy as np
    from eegio.base.objects.electrodes.elecs import Contacts

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
def eegts():
    """
    Creating a pytest fixture for a fake EEG Time series init.

    :return:
    """
    import numpy as np
    from eegio.base.objects.electrodes.elecs import Contacts
    from dev.dataset import EEGTimeSeries

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
    N = len(contacts)
    T = 2500
    rawdata = np.random.random((N, T))
    times = np.arange(T)
    samplerate = 1000
    modality = "ecog"

    eegts = EEGTimeSeries(rawdata, times, contacts, samplerate, modality)
    return eegts


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
    testdatadir = os.path.join(os.getcwd(), "./data/bids_layout/")
    filepath = os.path.join(testdatadir, "scalp_test.edf")
    return filepath


@pytest.fixture(scope="class")
def fif_fpath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/bids_layout/")
    filepath = os.path.join(testdatadir, "scalp_test_raw.fif")
    return filepath


@pytest.fixture(scope="class")
def test_arr():
    from eegio.base.utils.data_structures_utils import compute_samplepoints

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
