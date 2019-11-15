import pytest
import os
import tempfile

from eegio.base.objects.basebids import BaseBids
from eegio.base.objects.bids_patientv2 import BidsPatient

test_fpath = os.path.join(os.getcwd(), "./data/bids_layout/")


@pytest.mark.usefixture("edf_fpath")
@pytest.mark.usefixture("fif_fpath")
class Test_BidsLoader:
    """
    Testing class for loading in raw EEG data in the MNE framework.

    Ensures class type, and also correct functionality in annotations framework.
    """

    def test_bids_setup(self):
        # BIDS layout test data is in /data/bids_layout/
        layout = BaseBids(test_fpath, modality='ieeg')
        layout.print_summary()
        layout.print_dir_tree()

        # assert you can do so by creating arbitrary temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = BaseBids(tmpdir, modality='ieeg')
            layout.print_summary()
            layout.print_dir_tree()

            # there should be no data here when creating inside temporary directory
            assert layout.subject_nums == []

    def test_edf(self, edf_fpath):
        """
        Test loading in an edf filepath with lightweight wrapper of MNE reading of edf files.

        :param edf_fpath:
        :type edf_fpath:
        :return:
        :rtype:
        """
        bidspat = BidsPatient(subject_id='0001',
                              session_id='seizure',
                              datadir=test_fpath,
                              managing_user="test",
                              modality='eeg')

        # get sessions
        bidspat.get_subject_sessions()
        assert bidspat.numdatasets == 0
        assert len(bidspat.get_subject_sessions()) == 0

        # preprocess scalp EDF files
        bidspat.preprocess_edf_files([os.path.join(test_fpath, "scalp_test.edf"),
                                      os.path.join(test_fpath, "scalp_test_2.edf")])

