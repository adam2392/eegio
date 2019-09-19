import pytest

from eegio.base.objects.dataset.result_object import Result
from eegio.loaders.loader import Loader
from eegio.base.objects.elecs import Contacts


class TestResult:
    @pytest.mark.usefixture("result_fpath")
    def test_result_errors(self, result_fpath):
        """
        Test error and warnings raised by Result class.

        :param result: (Result)
        :return: None
        """
        pass

    @pytest.mark.usefixture("result_fpath")
    def test_result(self, result_fpath):
        """
        Test code runs without errors through all functions with dummy data.

        :param result:
        :return:
        """
        jsonfpath, npzfpath = result_fpath

        # load in the data
        loader = Loader(jsonfpath)
        datastruct, metadata = loader.read_npzjson(jsonfpath, npzfpath)

        assert isinstance(metadata, dict)
        assert datastruct.files
        # ensure data quality
        pertmats = datastruct["pertmats"]
        delvecs = datastruct["delvecs"]
        adjmats = datastruct["adjmats"]
        chlabels = metadata["chanlabels"]

        assert adjmats.ndim == 3
        assert pertmats.ndim == 2
        assert delvecs.ndim == 3

        # create a result
        sampletimes = metadata["samplepoints"]
        contacts = Contacts(chlabels, require_matching=False)
        resultobj = Result(pertmats, sampletimes, contacts, metadata=metadata)

        # assert resultobj is kosher
        assert resultobj.n_contacts == resultobj.get_data().shape[0]
        assert resultobj.length_of_result == resultobj.get_data().shape[1]

        # slice the data
