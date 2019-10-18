import pytest
import os
import tempfile
import numpy as np
from eegio.loaders import ResultLoader


class Test_ResultLoader:
    @pytest.mark.usefixture("result_fpath")
    def test_result_loading(self, result_fpath):
        jsonfpath, npzfpath = result_fpath

        # load in the data
        loader = ResultLoader(jsonfpath)
        resultobj = loader.load_file(npzfpath, jsonfpath=jsonfpath)
        original_data = resultobj.get_data().copy()

        with tempfile.TemporaryDirectory() as fdir:
            npyfile = os.path.join(fdir, "test.npy")
            np.save(npyfile, resultobj.get_data())

            resultobj = loader.load_file(npyfile, jsonfpath=jsonfpath)
            npydata = resultobj.get_data()
            # print(original_data.shape)
            # print(npydata.shape)
            # assert np.testing.assert_almost_equal(np.array(original_data), np.array(npydata))
