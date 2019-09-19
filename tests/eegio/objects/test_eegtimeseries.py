import numpy as np
import pytest
import datetime


@pytest.mark.usefixture("eegts")
class TestEEGTimeSeries:
    def test_eegts(self, eegts):
        """
        Test to test eeg time series object. 

        :param eegts: 
        :return: 
        """
        assert eegts.length_of_recording == len(eegts.times)
        assert eegts.n_contacts == eegts.mat.shape[0]
        assert eegts.info and isinstance(eegts.info, dict)
        if eegts.date_of_recording != None:
            assert isinstance(eegts.date_of_recording, datetime.datetime)
        # length in seconds computed should be approximately equal to length of recording
        pytest.approx(eegts.len_secs * eegts.samplerate, eegts.length_of_recording)

        # monopolar signal
        monopolar_signal = eegts.mat.copy()
        cavg_signal = eegts.mat - np.mean(eegts.mat, axis=0)

        # test common average
        eegts.set_common_avg_ref()

        # manually compute common average
        pytest.approx(monopolar_signal - cavg_signal, eegts.mat)

    def test_eegts_bipolar(self, eegts):
        # test bipolar
        eegts.set_bipolar()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_eegts_utility(self, eegts):
        # test filtering
        linefreq = 60
        eegts.filter_data(linefreq, bandpass_freq=[0.5, 300])

        # partition data into windows
        windowed_data = eegts.partition_into_windows(250, 125)
        assert len(windowed_data)

    def test_eegts_errors(self, eegts):
        """
        Test error and warnings raised by eegts class.

        :param eegts: (eegts)
        :return: None
        """
        # test filtering
        linefreq = 45
        with pytest.warns(UserWarning):
            eegts.filter_data(linefreq, bandpass_freq=[0.5, 300])

        # partition data into windows
        with pytest.warns(UserWarning):
            windowed_data = eegts.partition_into_windows(250, 1250)
        assert len(windowed_data)
