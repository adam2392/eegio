import sys
import os
import pytest

@pytest.mark.usefixtures("basebids")
class TestBaseBids:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def testbasebids(self, basebids):
        """

        Parameters
        ----------
        basebids : BaseBids

        Returns
        -------

        """
        assert os.path.exists(basebids.description_fpath)