import sys
import os

import pytest

@pytest.mark.usefixtures("bidsrun")
class TestBidsRun:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def testbidsrun(self, bidsrun):
        """

        Parameters
        ----------
        bidsrun : BidsRun

        Returns
        -------

        """
        pass