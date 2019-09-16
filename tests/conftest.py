""" Define fixtures available for eegio tests. """
import os
import os.path
from pathlib import Path

import pytest

RAWDATADIR = "/home/WIN/ali39/toshibaHDD/data/rawdata/"
RAWDATADIR = "/Users/adam2392/Downloads/tngpipeline/"
DATACENTERS = ['cleveland', 'jhu', 'nih', 'ummc']


@pytest.fixture(scope="function")
def contacts():
    """
    Creating a pytest fixture for a fake Contacts class init.

    :return:
    """
    import numpy as np
    from eegio.base.objects.elecs import Contacts
    contactlist = np.hstack(([f"A{i}" for i in range(16)],
                             [f"L{i}" for i in range(16)],
                             [f"B'{i}" for i in range(16)],
                             [f"D'{i}" for i in range(16)],
                             ["C'1", "C'2", "C'4", "C'8"],
                             ["C1", "C2", "C3", "C4", "C5", "C6"],
                             ))
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
    contactlist = np.hstack(([f"A{i}" for i in range(16)],
                             [f"L{i}" for i in range(16)],
                             [f"B'{i}" for i in range(16)],
                             [f"D'{i}" for i in range(16)],
                             ["C'1", "C'2", "C'4", "C'8"],
                             ["C1", "C2", "C3", "C4", "C5", "C6"],
                             ))
    contacts = Contacts(contactlist)
    N = len(contacts)
    T = 2500
    rawdata = np.random.random((N, T))
    times = np.arange(T)
    samplerate = 1000
    modality = 'ecog'

    eegts = EEGTimeSeries(rawdata, times, contacts, samplerate, modality)
    return eegts


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
    patient = Patient(patientid, datadir=datadir,
                      managing_user=managing_user, datasetids=datasetids)

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
    result = Patient(patientid, datadir=datadir,
                     managing_user=managing_user, datasetids=datasetids)

    return result

@pytest.fixture(scope="class")
def clinical_sheet():
    """
    Creating a pytest fixture for a fake Result data object.

    :return:
    """
    import numpy as np
    from eegio.base.objects.clinical import MasterClinicalSheet, PatientClinical

    patid = "test_patient_01"
    datasetids = [f"test_sz{i}" for i in range(2)]
    centerid = "jhu"
    example_datadict = {
        'length_of_recording': 400,
        'timepoints': np.hstack((np.arange(0, 100), np.arange(5, 105))),
    }

    patclin = PatientClinical(patid, datasetids, centerid)
    patclin.load_from_dict(example_datadict)
    clinical_sheet = MasterClinicalSheet([patclin])
    return clinical_sheet



@pytest.fixture(scope='class')
def edf_fpath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/")
    filepath = os.path.join(testdatadir, "scalp_test.edf")
    return filepath

@pytest.fixture(scope='class')
def fif_fpath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "./data/")
    filepath = os.path.join(testdatadir, "scalp_test_raw.fif")
    return filepath

@pytest.fixture(scope='class')
def ieeg_data_fif():
    """
    FOR TESTING FIF RAWDATA LOADING OF IEEG
    :return:
    """
    testdatadir = os.path.join(os.getcwd(), "tests/test_data/rawieeg/")
    testfilename = "ummc001_sz_3.json"

    return testdatadir, testfilename


@pytest.fixture(scope='class')
def seeg_data_fif():
    """
    FOR TESTING FIF RAWDATA LOADING OF IEEG
    :return:
    """
    testdatadir = os.path.join(os.getcwd(), "tests/test_data/rawieeg/")
    testfilename = "la02_sz.json"
    return testdatadir, testfilename


@pytest.fixture(scope='class')
def scalp_data_fif():
    """
    FOR TESTING FIF RAWDATA LOADING OF Scalp EEG Raw Data
    :return:
    """
    testdatadir = os.path.join(os.getcwd(), "tests/test_data/rawscalp/")
    testfilename = "scalp_test.json"
    return testdatadir, testfilename


@pytest.fixture(scope='class')
def result_frag_files():
    """
    Pytest fixture to get the filepaths for all the ltv data,
    so that tests can load and determine their quality.

    :return:
    """
    testdatadir = os.path.join(os.getcwd(), "tests/test_data/results/")
    testfilename = "pt1_sz_2_frag.json"
    return testdatadir, testfilename


@pytest.fixture(scope='class')
def all_data_fif():
    """
    Pytest fixture to get all the .fif/.json filepaths for the raw data,
    so that tests can load and determine their quality.

    :return:
    """
    # get all center directory names
    center_dirs = [d for d in os.listdir(
        RAWDATADIR) if os.path.isdir(os.path.join(RAWDATADIR, d))]

    # json and fiffilepaths
    fif_file_paths = []
    json_file_paths = []

    # go through each center and get the data
    for center in center_dirs:
        if center not in DATACENTERS:
            continue

        # get all the json/fif filepaths
        fif_paths = Path(os.path.join(RAWDATADIR, center)
                         ).glob("*/seeg/fif/*.fif")
        json_paths = Path(os.path.join(RAWDATADIR, center)
                          ).glob("*/seeg/fif/*.json")

        # pair edf name to infile path using a dictionary
        # edf_infiles_dict = {}
        for f in fif_paths:
            # get the actual dataset name
            fif_name = f.name
            fif_file_paths.append(os.path.join(
                os.path.dirname(str(f)), fif_name))

        for f in json_paths:
            json_name = f.name
            json_file_paths.append(os.path.join(
                os.path.dirname(str(f)), json_name))

    return json_file_paths


""" FOR CONF TEST OF EDM ALGORITHM - PARAMETER RANGES """
# def pytest_addoption(parser):
#     parser.addoption("--all", action="store_true",
#         help="run all combinations")
#
# def pytest_generate_tests(metafunc):
#     if 'param1' in metafunc.fixturenames:
#         if metafunc.config.getoption('all'):
#             end = 5
#         else:
#             end = 2
#         metafunc.parametrize("param1", range(end))
