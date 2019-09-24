import os
import tempfile

import numpy as np
import pytest

from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
from eegio.base.objects.elecs import Contacts
from eegio.loaders.loader import Loader
from eegio.writers.saveas import DataWriter


@pytest.mark.usefixture("clinical_fpath")
def test_preprocess_clinical(clinical_fpath):
    """
    First, load in data via edf.
    """
    pass
    # read_kwargs = {
    #     "fname": edf_fpath,
    #     "backend": "mne",
    #     "montage": None,
    #     "eog": None,
    #     "misc": None,
    # }
    # loader = Loader(edf_fpath, metadata={})
    # raw_mne, annotations = loader.read_edf(**read_kwargs)
    # metadata = raw_mne.metadata
    # chlabels = raw_mne.ch_names
    # samplerate = metadata["sfreq"]
    #
    # """
    # Second, convert to and object to save
    # """
    # modality = "scalp"
    # data, times = raw_mne.get_data(return_times=True)
    # contacts = Contacts(chlabels, require_matching=False)
    # eegts = EEGTimeSeries(data, times, contacts, samplerate, modality)
    #
    # # get matching montage and a list of bad channels
    # montage = eegts.get_best_matching_montage(chlabels)
    # bad_chans_list = eegts.get_bad_chs()
    #
    # """
    # Third, save as fif file
    # """
    # # create the metadata data struct
    # metadata["bads"] = bad_chans_list
    # metadata["montage"] = montage
    #
    # writer = DataWriter()
    # with tempfile.TemporaryDirectory() as fdir:
    #     fpath = os.path.join(fdir, "test_raw.fif")
    #     rawmne = writer.saveas_fif(fpath, eegts.get_data(), eegts.metadata)
    #     np.testing.assert_array_equal(rawmne.get_data(), eegts.get_data())
    #     np.testing.assert_array_equal(rawmne.get_data(), raw_mne.get_data())
    #
    #     # load in fif file
    #     raw_mne, annotations = loader.read_fif(fpath)
    #     metadata = raw_mne.metadata
    #     chlabels = raw_mne.ch_names
