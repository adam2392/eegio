""" Define fixtures available for eegio tests. """
import os
import os.path
from pathlib import Path

import pytest

RAWDATADIR = "/home/WIN/ali39/toshibaHDD/data/rawdata/"
RAWDATADIR = "/Users/adam2392/Downloads/tngpipeline/"
DATACENTERS = ['cleveland', 'jhu', 'nih', 'ummc']


@pytest.fixture(scope='class')
def edffilepath():
    """
    FOR TESTING EDF RAWDATA OF PREFORMATTING INTO FIF+JSON PAIRS

    :return:
    """
    # load in edf data
    testdatadir = os.path.join(os.getcwd(), "tests/test_data/rawieeg/")
    testfilename = os.path.join(testdatadir, "pt1sz2.edf")
    return testfilename


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
    testfilename = "pt1_sz_1.json"
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


# @pytest.fixture(scope='class')
# def result_freq_files():
#     """
#     Pytest fixture to get the filepaths for all the frequency files data,
#     so that tests can load and determine their quality.
#
#     :return:
#     """
#     testdatadir = os.path.join(os.getcwd(), "tests/test_data/results/")
#     testfilename = "pt1_sz_2_freq.json"
#     return testdatadir, testfilename


# @pytest.fixture(scope='class')
# def result_hdf_data():
#     """
#     Pytest fixture to get the filepaths for all the ltv data,
#     so that tests can load and determine their quality.
#
#     :return:
#     """
#     testdatadir = os.path.join(os.getcwd(), "tests/test_data/results/")
#     testfilename = "pt1_fragresults.hd5"
#     return testdatadir, testfilename


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
