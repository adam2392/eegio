import pytest

from eegio.base.objects.dataset.result_object import Result
from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
from eegio.base.objects.elecs import Contacts
from eegio.loaders.loader import Loader


@pytest.mark.usefixture("result_fpath")
def test_load_resultdata(result_fpath):
    """
    Test error and warnings raised by Result class.

    :param result: (Result)
    :return: None
    """
    """
    First, load in data via np.
    """
    result_fpath, result_npzfpath = result_fpath
    loader = Loader(result_fpath)

    pass
    """
    Second, convert to and object
    """
    # modality = "scalp"
    # rawdata, times = raw_mne.get_data(return_times=True)
    # contacts = Contacts(chlabels, require_matching=False)
    # result = Result(rawdata, times, contacts, samplerate, modality)
