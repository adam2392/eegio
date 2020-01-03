import os
import tempfile
import mne
import mne_bids
from eegio.loaders import BaseBids
from eegio.loaders import BidsRun
from eegio.loaders import BidsPatient
from mne_bids import make_bids_basename

test_fpath = os.path.join(os.getcwd(), "./data/bids_layout/")
import pytest


@pytest.mark.usefixture("edf_fpath")
@pytest.mark.usefixture("fif_fpath")
@pytest.mark.usefixture("tmp_bids_root")
class Test_BidsLoader:
    """
    Testing class for loading in raw EEG data in the MNE framework.

    Ensures class type, and also correct functionality in annotations framework.
    """

    def test_bids_setup(self, tmp_bids_root):
        # BIDS layout test data is in /data/bids_layout/
        layout = BaseBids(bids_root=tmp_bids_root)
        layout.print_summary()
        layout.print_dir_tree()

        # assert you can do so by creating arbitrary temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = BaseBids(bids_root=tmpdir)
            layout.print_summary()
            layout.print_dir_tree()

            # there should be no data here when creating inside temporary directory
            assert layout.subject_nums == []

    def test_bids_run_setup(self, edf_fpath, tmp_bids_root):
        """ TODO: make work with temporary directory. """
        pass
        # test_subjectid = "0001"
        # session_id = "seizure"
        # kind = "eeg"
        # run_id = "00"
        # bids_fname = make_bids_basename(
        #             subject=test_subjectid,
        #             session=session_id,
        #             run=run_id,
        #             suffix="eeg.edf",
        #         )
        # if os.path.exists(
        #     os.path.join(
        #         test_fpath,
        #         bids_fname
        #     )
        # ):
        #     run = BidsRun(tmp_bids_root=tmp_bids_root, bids_fname=bids_fname)
        #     run._create_bidsrun(edf_fpath)
        # else:
        #     run = BidsRun(tmp_bids_root=tmp_bids_root, bids_fname=bids_fname)
        #     run._create_bidsrun(edf_fpath)

    @pytest.mark.skip(reason="TODO: dev")
    def test_edf(self, edf_fpath):
        """
        Test loading in an edf filepath with lightweight wrapper of MNE reading of edf files.

        :param edf_fpath:
        :type edf_fpath:
        :return:
        :rtype:
        """
        test_subjectid = "0001"
        bidspat = BidsPatient(None, bids_fname="eeg")
        # get sessions
        bidspat.get_subject_sessions()
        assert bidspat.numdatasets >= 0
        assert len(bidspat.get_subject_sessions()) >= 0

        # preprocess scalp EDF files
        bidspat.preprocess_edf_files(
            [
                os.path.join(test_fpath, "scalp_test.edf"),
                os.path.join(test_fpath, "scalp_test_2.edf"),
            ]
        )

        # should be able to read in all the data related to the edf file
        test_modality = "eeg"
        session_ids = bidspat.get_subject_sessions()
        for session_id in session_ids:
            run_ids = bidspat.get_subject_runs()

            for run_id in run_ids:
                rawmne = bidspat.load_run(
                    session_id=session_id, run_id=run_id, kind=test_modality
                )

                assert isinstance(rawmne, mne.io.BaseRaw)

        # load a specific dataset in mind and allow to pass in filename
        test_filename = mne_bids.make_bids_basename(
            subject=test_subjectid, session=session_id, run=run_id
        )
        # should not pass in filename and other variables
        with pytest.raises(Exception):
            rawmne, metadata = bidspat.load_run(filename=test_filename)

        # should be able to load a dataset based on its BIDS
        bidspat.load_run(session_id=session_id, run_id=run_id, kind=test_modality)
