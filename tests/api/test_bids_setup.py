import mne_bids
import pytest
from mne_bids import make_bids_basename

from eegio.base.utils.bids_helper import BidsBuilder, BidsUtils, BidsConverter


@pytest.mark.usefixture(["bids_dir", "edf_fpath"])
class TestBidsPatient:
    def test_bids_setup(self, bids_dir, edf_fpath):
        """
        Integration test for setting up a bids directory from scratch. Should test:

        1. Initial setup
        2. loading runs
        3. loading patients

        Parameters
        ----------
        bids_dir :

        Returns
        -------

        """
        # create the BIDS dirctory structure
        test_subjectid = "0002"
        modality = "eeg"
        test_sessionid = "seizure"
        test_runid = "01"
        # bids_dir = os.path.join(os.getcwd(), "./data/bids_layout/")
        BidsBuilder.make_bids_folder_struct(
            datadir=bids_dir,
            subject=test_subjectid,
            kind=modality,
            session_id=test_sessionid,
        )
        # add a bids run
        bids_basename = make_bids_basename(
            subject=test_subjectid, session=test_sessionid, run=test_runid
        )
        # call bidsbuilder pipeline
        bids_path = BidsConverter.convert_to_bids(
            edf_fpath=edf_fpath,
            bids_dir=bids_dir,
            bids_basename=bids_basename,
            overwrite=True,
        )
        # convert this into a fif format
        edf_fname = bids_basename
        bids_path = BidsConverter.preprocess_into_fif(edf_fname, bids_path)

        # load it in using mne_bids again
        bids_basename = (
            make_bids_basename(
                subject=test_subjectid,
                session=test_sessionid,
                processing="fif",
                run=test_runid,
                suffix="raw_eeg",
            )
            + ".fif"
        )
        raw = mne_bids.read_raw_bids(bids_basename, bids_path)

        """
        Test
        """

    @pytest.mark.skip()
    def test_bids_setup_errors(self):
        """
        Test error and warnings raised by a pipeline used for Bids Setup
        that should be raised when a user does something wrong.

        Parameters
        ----------

        Returns
        -------
        """
        pass

    @pytest.mark.skip()
    def test_bids_setup_catastrophic(self):
        pass
