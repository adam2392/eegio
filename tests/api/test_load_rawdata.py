import pytest

from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
from eegio.base.objects.electrodes.elecs import Contacts
from eegio.loaders.loader import Loader


@pytest.mark.usefixture("fif_fpath")
def test_load_rawdata_mne(fif_fpath):
    """
    First, load in data via fif.
    """
    loader = Loader(fif_fpath)
    # load in fif file
    raw_mne, annotations = loader.read_fif(fif_fpath, return_mne=True)
    info = raw_mne.info
    chlabels = raw_mne.ch_names
    samplerate = info["sfreq"]

    """
    Second, convert to and object
    """
    modality = "scalp"
    rawdata, times = raw_mne.get_data(return_times=True)
    contacts = Contacts(chlabels, require_matching=False)
    eegts = EEGTimeSeries(rawdata, times, contacts, samplerate, modality)

    assert isinstance(eegts, EEGTimeSeries)


@pytest.mark.usefixture("fif_fpath")
def test_load_rawdata_eegts(fif_fpath):
    """
    First, load in data via fif automatically to eegts
    """
    loader = Loader(fif_fpath)
    # load in fif file
    eegts = loader.read_fif(fif_fpath)

    assert isinstance(eegts, EEGTimeSeries)
