import mne
import numpy as np
import pytest

from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
from eegio.base.objects.elecs import Contacts
from eegio.format.scrubber import ChannelScrub
from eegio.loaders.loader import Loader


@pytest.mark.usefixture("edf_fpath")
@pytest.mark.usefixture("fif_fpath")
class Test_Loader:
    """
    Testing class for loading in raw EEG data in the MNE framework.

    Ensures class type, and also correct functionality in annotations framework.
    """

    def test_load_file(self, edf_fpath, fif_fpath):
        loader = Loader(edf_fpath, metadata={})

        eegts = loader.load_file(edf_fpath)

        eegts = loader.load_file(fif_fpath)

    def test_edf(self, edf_fpath):
        """
        Test loading in an edf filepath with lightweight wrapper of MNE reading of edf files.

        :param edf_fpath:
        :type edf_fpath:
        :return:
        :rtype:
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
        filesize = loader.get_size()

        loader.update_metadata(filesize=filesize)
        metadata = loader.get_metadata()
        assert np.isclose(filesize, 4.1, atol=0.5)

        raw_mne, annotations = loader.read_edf(**read_kwargs)
        info = raw_mne.info
        chlabels = raw_mne.ch_names
        n_times = raw_mne.n_times
        print(info)
        print(annotations)

        assert len(chlabels) == info["nchan"]
        assert info["line_freq"]
        assert isinstance(raw_mne, mne.io.BaseRaw)
        assert raw_mne.get_data(stop=50).shape[0] == len(chlabels)

        read_kwargs = {
            "fname": edf_fpath,
            "backend": "mne",
            "montage": None,
            "eog": None,
            "misc": None,
        }
        loader = Loader(edf_fpath, metadata={})
        filesize = loader.get_size()
        loader.update_metadata(filesize=filesize)
        metadata = loader.get_metadata()
        assert np.isclose(filesize, 4.1, atol=0.5)
        eegts = loader.read_edf(**read_kwargs)

        assert isinstance(eegts, EEGTimeSeries)

    def test_edf_pyedflib(self, edf_fpath):
        pass

    def test_fif(self, fif_fpath):
        """
        Test explicit fif loading.

        :param fif_fpath:
        :type fif_fpath:
        :return:
        :rtype:
        """
        read_kwargs = {"fname": fif_fpath, "linefreq": 60, "return_mne": True}
        loader = Loader(fif_fpath, metadata={})
        raw_mne, annotations = loader.read_fif(**read_kwargs)
        info = raw_mne.info
        chlabels = raw_mne.ch_names
        n_times = raw_mne.n_times

        print(raw_mne)
        print(annotations)
        assert info["meas_date"]
        assert isinstance(raw_mne, mne.io.BaseRaw)
        assert len(chlabels) == info["nchan"]
        assert info["line_freq"]

        read_kwargs = {"fname": fif_fpath, "linefreq": 60}
        loader = Loader(fif_fpath, metadata={})
        eegts = loader.read_fif(**read_kwargs)

        assert isinstance(eegts, EEGTimeSeries)

    def test_create_metadata(self):
        pass

    def test_convert_fif(self):
        pass

    def test_fif_with_scalp_eegts(self, fif_fpath):
        """
        Integration test to load fif raw file (from MNE) that is a scalp EEG dataset, and transform into an EEGTimeSeries
        object.

        Integrates scrubbing at the channel level, and utility functions built in for EEGTS to ensure every inner-obj
        handling is done correctly.

        :param fif_fpath:
        :type fif_fpath:
        :return:
        :rtype:
        """
        # load in fif file
        read_kwargs = {"fname": fif_fpath, "linefreq": 60, "return_mne": True}
        loader = Loader(fif_fpath, metadata={})
        raw_mne, annotations = loader.read_fif(**read_kwargs)
        info = raw_mne.info
        chlabels = raw_mne.ch_names
        n_times = raw_mne.n_times

        # since passing in scalp data, raise error w/o using requirematching
        with pytest.raises(ValueError):
            contacts = Contacts(chlabels)

        # create intermediate data structures
        raw_mne.load_data()
        raw_mne = ChannelScrub.channel_text_scrub(raw_mne)
        chlabels = raw_mne.ch_names
        badchs = ChannelScrub.look_for_bad_channels(chlabels)
        goodinds = [i for i, ch in enumerate(chlabels) if ch not in badchs]
        raw_mne.drop_channels(badchs)
        chlabels = raw_mne.ch_names

        # load in the cleaned ch labels
        contacts = Contacts(chlabels, require_matching=False)
        samplerate = info["sfreq"]
        modality = "scalp"
        rawdata, times = raw_mne.get_data(return_times=True)

        # create EEG TS object
        eegts = EEGTimeSeries(rawdata, times, contacts, samplerate, modality)

        # test getting the "best montage" wrapping MNE
        testchs = eegts.chanlabels.copy()
        print(testchs)
        best_montage = eegts.get_best_matching_montage(testchs)
        # get the indices that fit a montage in scalp EEG
        montage_inds = eegts.get_montage_channel_indices(best_montage, testchs)

        assert len(montage_inds) > 0

        print(testchs)
        # naturally sort contacts
        eegts.natsort_contacts()

        # trim dataset in time and by electrode
        test_getch = "fp1"
        trimmed_rawdata_time = eegts.trim_dataset()
        trimmed_rawdata_chs = eegts.get_channel_data(test_getch)

        assert trimmed_rawdata_chs.ndim == 1
        assert trimmed_rawdata_time.ndim == 2

        # remove channels
        test_remove = "fp1"
        eegts.remove_channels(test_remove)
        assert len(eegts.chanlabels) == len(testchs) - 1
        assert eegts.mat.shape[0] == eegts.ncontacts

        # restore to original state
        eegts.reset()

        assert eegts.mat.shape[0] == eegts.ncontacts
        assert eegts.ncontacts == len(testchs)
