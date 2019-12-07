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
        modality = "ieeg"
        with tempfile.TemporaryDirectory() as datadir:
            basebids = BaseBids(datadir=datadir, modality=modality)
            assert os.path.exists(basebids.description_fpath)
