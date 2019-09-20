import pytest

from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
from eegio.base.objects.elecs import Contacts
from eegio.loaders.loader import Loader


@pytest.mark.usefixture("fif_fpath")
def test_load_rawdata(fif_fpath):
    """
    Test error and warnings raised by Result class.

    :param result: (Result)
    :return: None
    """
    """
    First, load in data via fif.
    """
    loader = Loader(fif_fpath)
    # load in fif file
    raw_mne, annotations = loader.read_fif(fif_fpath)
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
