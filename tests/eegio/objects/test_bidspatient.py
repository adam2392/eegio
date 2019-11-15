import sys

import pytest

from eegio.base.config import MB

@pytest.mark.usefixture("bidsPatient")
class TestBidsPatient:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_bidsPatient(self, bidsPatient):
        """
        Test to test BidsPatient object.

        Parameters
        ----------
        bidsPatient: BidsPatient
            The class object to test

        Returns
        -------
        """
        bidspatient = bidspatient.create_fake_example()

        assert bidspatient.numdatasets >= 0

        fpaths = bidspatient.get_all_filepaths()
        assert len(fpaths) == length(bidspatient.datasetids)

    def test_bidsPatient_channelfunc(self, bidsPatient):
        assert len(bidsPatient.wm_channels) >= 0
        assert len(bidsPatient.bad_channels) >= 0

        # test working w/ channels
        test_bad_ch = ["a1", "a2", "a3"]
        test_wm_ch = ["b1", "b3"]
        
        bidsPatient.add_bad_channels(test_bad_ch)
        bidsPatient.add_wm_channels(test_wm_channels)

        assert len(bidsPatient.wm_channels) == len(test_wm_ch)
        assert len(bidsPatient.bad_channels) == len(test_bad_ch)

        # make sure that bidspatient object isnt loading in any explicit data
        # into the object
        assert sys.getsizeof(bidspatient) < 5 * MB

    def test_bidsPatient_errors(self, bidsPatient):
        """
        Test error and warnings raised by patient class.

        Parameters
        ----------
        bidsPatient: BidsPatient
            The class object to test

        Returns
        -------
        """
        # partition data into windows
        with pytest.warns(RuntimeWarning):
            fpaths = bidsPatient.get_all_filepaths()


        
