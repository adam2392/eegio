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

        bidspatient.create_run(TEST_DATA_DIR)

        assert bidspatient.numdatasets == 1

        channel_data = bidspatient.loader.load_scans_tsv()
        channel_data.to_csv('test_channels.csv')
