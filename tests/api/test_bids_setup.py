import os

import mne_bids
import pytest
from mne_bids import make_bids_basename, make_bids_folders

from eegio.base.utils.bids_helper import BidsConverter


@pytest.mark.usefixture(["tmp_bids_root", "edf_fpath"])
class TestBidsPatient:
    def test_bids_setup(self, tmp_bids_root, edf_fpath):
        """
        Integration test for setting up a bids directory from scratch.

        Should test:
        1. Initial setup
        2. loading runs
        3. loading patients

        Parameters
        ----------
        bids_root :

        Returns
        -------

        """
        # create the BIDS dirctory structure
        test_subjectid = "0002"
        modality = "eeg"
        test_sessionid = "seizure"
        test_runid = "01"
        task = "monitor"
        line_freq = 60
        # create the BIDS directory structure
        if not os.path.exists(tmp_bids_root):
            print("Making bids root directory.")
            make_bids_folders(
                output_path=tmp_bids_root,
                session=test_sessionid,
                subject=test_subjectid,
                kind=modality,
            )
        # add a bids run
        bids_basename = make_bids_basename(
            subject=test_subjectid,
            acquisition=modality,
            session=test_sessionid,
            run=test_runid,
            task=task,
        )

        # call bidsbuilder pipeline
        tmp_bids_root = BidsConverter.convert_to_bids(
            edf_fpath=edf_fpath,
            bids_root=tmp_bids_root,
            bids_basename=bids_basename,
            line_freq=line_freq,
            overwrite=True,
        )

        # load it in using mne_bids again
        bids_fname = bids_basename + f"_{modality}.fif"
        raw = mne_bids.read_raw_bids(bids_fname, tmp_bids_root)

    @pytest.mark.skip()
    def test_bids_setup_errors(self):
        """
        Test error and warnings raised by a pipeline used for Bids Setup.

        Errors should be raised when a user does something wrong.

        Parameters
        ----------

        Returns
        -------
        """
        pass

    @pytest.mark.skip()
    def test_bids_setup_catastrophic(self):
        pass
