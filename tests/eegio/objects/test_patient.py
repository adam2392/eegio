import numpy as np
import pytest


@pytest.mark.usefixture('patient')
class TestPatient():
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_patient(self, patient):
        """
        Test to test Patient object.

        :param patient:
        :return:
        """
        patient = patient.create_fake_example()

        assert patient.numdatasets >= 0

        fpaths = patient.get_all_filepaths()
        assert len(fpaths) == len(patient.datasetids)

    def test_patient_channelfunc(self, patient):
        assert len(patient.wm_channels) >= 0
        assert len(patient.bad_channels) >= 0

        # test working w/ channels
        test_bad_ch = ["a1", "a2", "a3"]
        test_wm_ch = ["b1", "b3"]

        patient.add_bad_channels(test_bad_ch)
        patient.add_wm_channels(test_wm_ch)

        assert len(patient.wm_channels) == len(test_wm_ch)
        assert len(patient.bad_channels) == len(test_bad_ch)

    def test_patient_errors(self, patient):
        """
        Test error and warnings raised by patient class.

        :param patient:
        :return: None
        """
        # partition data into windows
        with pytest.warns(RuntimeWarning):
            fpaths = patient.get_all_filepaths()
