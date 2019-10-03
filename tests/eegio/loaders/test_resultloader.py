import pytest

from eegio.loaders import ResultLoader


class Test_ResultLoader():
    @pytest.mark.usefixture("result_fpath")
    def test_result_loading(self, result_fpath):
        jsonfpath, npzfpath = result_fpath

        # load in the data
        loader = ResultLoader(jsonfpath)
        resultobj = loader.load_file(npzfpath, jsonfpath=jsonfpath)
