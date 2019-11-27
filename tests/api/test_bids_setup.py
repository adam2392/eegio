import os
import tempfile
import mne
import pytest

from eegio.base.objects.bids.bids_patient import BidsPatient


@pytest.mark.usefixture("bids_dir")
class TestBidsPatient:
    def test_bids_setup(self, bids_dir):
        """
        Integration test for setting up a bids directory from scratch. Should test:

        1. Initial setup
        2. loading runs
        3. loading patients

        Parameters
        ----------
        bids_dir :

        Returns
        -------

        """
        pass

    @pytest.mark.skip()
    def test_bids_setup_errors(self):
        """
        Test error and warnings raised by a pipeline used for Bids Setup
        that should be raised when a user does something wrong.

        Parameters
        ----------

        Returns
        -------
        """
        pass

    @pytest.mark.skip()
    def test_bids_setup_catastrophic(self):
        pass