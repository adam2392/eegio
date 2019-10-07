import pytest

from eegio.loaders import ResultLoader


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
    loader = ResultLoader(result_fpath)

    pass
    """
    Second, convert to and object
    """
    # modality = "scalp"
    # data, times = raw_mne.get_data(return_times=True)
    # contacts = Contacts(chlabels, require_matching=False)
    # result = Result(data, times, contacts, samplerate, modality)
