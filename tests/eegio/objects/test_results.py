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
        model_attributes = {
            "winsize": metadata["winsize"],
            "stepsize": metadata["stepsize"],
            "samplerate": metadata["samplerate"],
        }
        contacts = Contacts(chlabels, require_matching=False)
        resultobj = Result(
            pertmats,
            sampletimes,
            contacts,
            metadata=metadata,
            model_attributes=model_attributes,
        )

        # slice the data
        samplepoints = resultobj.samplepoints
        # assert resultobj is kosher
        assert resultobj.ncontacts == resultobj.get_data().shape[0]
        assert len(samplepoints) == pertmats.shape[1]
        assert len(resultobj.timepoints) == len(samplepoints)
        assert len(samplepoints) == len(resultobj)

        # drop certain channels
        beforelen = len(contacts)
        resultobj.mask_channels(contacts[0:3])
        assert resultobj.ncontacts == beforelen - 3

        # reset and things should be back to normal
        resultobj.reset()
        assert beforelen == resultobj.ncontacts
