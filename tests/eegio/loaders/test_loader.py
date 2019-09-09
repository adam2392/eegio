import mne
import pytest

from eegio.loaders.loader import Loader


# @pytest.mark.skip(reason="Already tested as part of rawedf to results!")
@pytest.mark.usefixture('edf_fpath')
@pytest.mark.usefixture('fif_fpath')
class Test_Loader():
    """
    Testing class for loading in raw EEG data in the MNE framework.

    Ensures class type, and also correct functionality in annotations framework.
    """

    def test_edf(self, edf_fpath):
        read_kwargs = {
            "fname": edf_fpath,
            "backend": "mne",
            "montage": None,
            "eog": None,
            "misc": None,
        }
        loader = Loader(edf_fpath, metadata={})
        raw_mne, annotations = loader.read_edf(**read_kwargs)
        info = raw_mne.info
        chlabels = raw_mne.ch_names
        n_times = raw_mne.n_times
        print(info)
        print(annotations)

        assert len(chlabels) == info['nchan']
        assert info['line_freq']
        assert (isinstance(raw_mne, mne.io.BaseRaw))
        assert raw_mne.get_data(stop=50).shape[0] == len(chlabels)

    def test_edf_pyedflib(self, edf_fpath):
        pass

    def test_fif(self, fif_fpath):
        read_kwargs = {
            "fname": fif_fpath,
            "linefreq": 60,
        }
        loader = Loader(fif_fpath, metadata={})
        raw_mne, annotations = loader.read_fif(**read_kwargs)
        info = raw_mne.info
        chlabels = raw_mne.ch_names
        n_times = raw_mne.n_times

        print(raw_mne)
        print(annotations)
        assert (isinstance(raw_mne, mne.io.BaseRaw))
        assert len(chlabels) == info['nchan']
        assert info['line_freq']

    def test_create_metadata(self):
        pass

    def test_convert_fif(self):
        pass

    def test_convert_mat(self):
        pass
