import os
import tempfile
import pytest

from eegio.loaders.bids import BaseBids


class TestBaseBids:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def testbasebids(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        # check that base bids initializes correct class attributes
        with tempfile.TemporaryDirectory() as bids_root:
            basebids = BaseBids(bids_root=bids_root)
            assert os.path.exists(basebids.description_fpath)
