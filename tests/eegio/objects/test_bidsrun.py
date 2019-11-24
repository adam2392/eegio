import sys
import os

import pytest

import pandas as pd

from eegio.base.config import TEST_DATA_DIR

@pytest.mark.usefixtures("bidspatient")
class TestBidsRun:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def testbidsrun(self, bidspatient):
        """

        Parameters
        ----------
        bidspatient : BidsPatient

        Returns
        -------

        """
        assert bidspatient.numdatasets == 0
        assert bidspatient.num_subjects == 0

        scans = ['/Users/patrick/Desktop/eegio/data/bids_layout/scalp_test.edf']
        bidsrun = bidspatient.add_scans(scans)

        assert bidspatient.numdatasets == 1
        assert bidspatient.num_subjects == 1
