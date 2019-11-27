import pytest
import os
import tempfile
import mne
import mne_bids
from eegio.base.objects.bids.basebids import BaseBids
from eegio.base.objects.bids.bids_run import BidsRun
from eegio.base.objects.bids.bids_patient import BidsPatient

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

    def test_bids_run_setup(self, edf_fpath):
        pass
        test_subjectid = '0001'
        session_id = "seizure"
        kind = "eeg"
        run_id = "00"
        run = BidsRun(subject_id=test_subjectid,
                      session_id=session_id,
                      run_id=run_id,
                      datadir=test_fpath,
                      modality=kind)
        run._create_bidsrun(edf_fpath)

    def test_edf(self, edf_fpath):
        """
        Test loading in an edf filepath with lightweight wrapper of MNE reading of edf files.

        :param edf_fpath:
        :type edf_fpath:
        :return:
        :rtype:
        """
        test_subjectid = '0001'
        bidspat = BidsPatient(subject_id=test_subjectid,
                              session_id='seizure',
                              datadir=test_fpath,
                              managing_user="test",
                              modality='eeg')
        # get sessions
        bidspat.get_subject_sessions()
        assert bidspat.numdatasets >= 0
        assert len(bidspat.get_subject_sessions()) >= 0

        # preprocess scalp EDF files
        bidspat.preprocess_edf_files([os.path.join(test_fpath, "scalp_test.edf"),
                                      os.path.join(test_fpath, "scalp_test_2.edf")])

        # should be able to read in all the data related to the edf file
        test_modality = 'eeg'
        session_ids = bidspat.get_subject_sessions()
        for session_id in session_ids:
            run_ids = bidspat.get_subject_runs(sesion_id=session_id)

            for run_id in run_ids:
                rawmne, metadata = bidspat.load_dataset(session_id=session_id, run_id=run_id, kind=test_modality)

                assert isinstance(rawmne, mne.io.Raw)

        # load a specific dataset in mind and allow to pass in filename
        test_filename = mne_bids.make_bids_basename(subject=test_subjectid, session=session_id, run=run_id)
        # should not pass in filename and other variables
        with pytest.raises(Exception):
            bidspat.load_dataset(filename=test_filename, session_id=session_id, run_id=run_id, kind=test_modality)

        # should be able to load a dataset based on its filename
        rawmne, metadata = bidspat.load_dataset(filename=test_filename)


