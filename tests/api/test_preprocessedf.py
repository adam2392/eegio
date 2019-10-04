import json
import os
import tempfile

import mne
import numpy as np
import pandas as pd
import pytest

import eegio
from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
from eegio.base.objects.elecs import Contacts
from eegio.loaders.loader import Loader
from eegio.writers import DataWriter


@pytest.mark.usefixture("edf_fpath")
def test_preprocess_edf(edf_fpath):
    """
    Test error and warnings raised by Result class.

    Parameters
    ----------
    edf_fpath :

    Returns
    -------

    """
    """
    First, load in data via edf.
    """
    read_kwargs = {
        "fname": edf_fpath,
        "backend": "mne",
        "montage": None,
        "eog": None,
        "misc": None,
        "return_mne": True,
    }
    loader = Loader(edf_fpath, metadata={})
    raw_mne, annotations = loader.read_edf(**read_kwargs)
    info = raw_mne.info
    chlabels = raw_mne.ch_names
    samplerate = info["sfreq"]

    """
    Second, convert to and object to save
    """
    modality = "scalp"
    rawdata, times = raw_mne.get_data(return_times=True)
    contacts = Contacts(chlabels, require_matching=False)
    eegts = EEGTimeSeries(rawdata, times, contacts, samplerate, modality)

    # get matching montage and a list of bad channels
    montage = eegts.get_best_matching_montage(chlabels)
    bad_chans_list = eegts.get_bad_chs()

    """
    Third, save as fif file
    """
    # create the metadata data struct
    info["bads"] = bad_chans_list
    info["montage"] = montage

    writer = DataWriter()
    with tempfile.TemporaryDirectory() as fdir:
        fpath = os.path.join(fdir, "test_raw.fif")
        rawmne = writer.saveas_fif(fpath, eegts.get_data(), eegts.info)
        np.testing.assert_array_equal(rawmne.get_data(), eegts.get_data())
        np.testing.assert_array_equal(rawmne.get_data(), raw_mne.get_data())

        # save the annotations
        pass

        # load in fif file
        raw_mne, annotations = loader.read_fif(fpath, return_mne=True)
        info = raw_mne.info
        chlabels = raw_mne.ch_names


@pytest.mark.usefixture("edf_fpath")
@pytest.mark.usefixture("clinical_fpath")
def test_preprocess_format_api(edf_fpath, clinical_fpath):
    """
    Tests preprocessing an edf file into a formatted .fif + .json pair of files.

    Parameters
    ----------
    edf_fpath :
    clinical_fpath :

    Returns
    -------

    """
    writer = DataWriter()

    with tempfile.TemporaryDirectory() as fdir:
        temp_outfpath = os.path.join(fdir, "test_raw.fif")
        temp_jsonfpath = os.path.join(fdir, "test_raw.json")
        raw, metadata = eegio.format_eegdata(edf_fpath, temp_outfpath, temp_jsonfpath)

        # assert saved mne file and json file can load in
        raw = mne.io.read_raw_fif(temp_outfpath, preload=True)
        with open(temp_jsonfpath, "r", encoding="utf8") as fp:
            metadata = json.load(fp)

        assert isinstance(metadata, dict)

        temp_outfpath = os.path.join(fdir, "test_clinical.csv")
        formatted_clinsheet = eegio.format_clinical_sheet(
            clinical_fpath, cols_to_reg_expand=["bad_channels"]
        )

        # assert saved mne file and json file can load in
        formatted_clinsheet.to_csv(temp_outfpath, index=None)

        test_csv = pd.read_csv(temp_outfpath, index_col=None)
