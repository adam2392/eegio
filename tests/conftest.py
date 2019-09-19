""" Define fixtures available for eegio tests. """
import os
import os.path

import numpy as np
import pytest


@pytest.fixture(scope="function")
def contacts():
    """
    Creating a pytest fixture for a fake Contacts class init.

    :return:
    """
    import numpy as np
    from eegio.base.objects.elecs import Contacts

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
    from eegio.base.objects.elecs import Contacts
    from eegio.base.objects.dataset.eegts_object import EEGTimeSeries

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
def ieegts():
    """
    Creating a pytest fixture for a fake EEG Time series init.

    :return:
    """
    ieegts = []
    return ieegts


@pytest.fixture(scope="class")
def patient():
    """
    Creating a pytest fixture for a fake Patient data object.

    :return:
    """
    from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
    from eegio.base.objects.patient import Patient

    eegts = EEGTimeSeries.create_fake_example()

    patientid = "test_patient_01"
    datasetids = [f"test_sz{i}" for i in range(5)]
    managing_user = "EZTrack"
    datadir = os.path.join("./")
    patient = Patient(
        patientid, datadir=datadir, managing_user=managing_user, datasetids=datasetids
    )

    return patient


@pytest.fixture(scope="class")
def result():
    """
    Creating a pytest fixture for a fake Result data object.

    :return:
    """
    from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
    from eegio.base.objects.patient import Patient

    eegts = EEGTimeSeries.create_fake_example()

    patientid = "test_patient_01"
    datasetids = [f"test_sz{i}" for i in range(5)]
    managing_user = "EZTrack"
    datadir = os.path.join("./")
    result = Patient(
        patientid, datadir=datadir, managing_user=managing_user, datasetids=datasetids
    )

    return result


@pytest.fixture(scope="class")
def clinical_sheet():
    """
    Creating a pytest fixture for a fake Result data object.

    :return:
    """
    import numpy as np
    from eegio.base.objects.clinical import PatientClinical

    patid = "test_patient_01"
    datasetids = [f"test_sz{i}" for i in range(2)]
    centerid = "jhu"
    example_datadict = {
        "length_of_recording": 400,
        "timepoints": np.hstack((np.arange(0, 100), np.arange(5, 105))),
    }

    patclin = PatientClinical(patid, datasetids, centerid)
    patclin.load_from_dict(example_datadict)
    # clinical_sheet = MasterClinicalSheet([patclin])
    # return clinical_sheet


@pytest.fixture(scope="class")
def edf_fpath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/")
    filepath = os.path.join(testdatadir, "scalp_test.edf")
    return filepath


@pytest.fixture(scope="class")
def fif_fpath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/")
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


@pytest.fixture(scope="class")
def rawio():
    """
    FOR TESTING FIF RAWDATA LOADING OF IEEG
    :return:
    """
    rawio = []
    return rawio
