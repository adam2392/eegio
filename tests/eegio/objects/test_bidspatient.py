import sys

import pytest

import pandas as pd

from eegio.base.config import MB

@pytest.mark.usefixture("bidspatient")
class TestBidsPatient:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_bidspatient(self, bidspatient):
        """
        Test to test BidsPatient object.

        Parameters
        ----------
        bidspatient: Bidspatient
            The class object to test

        Returns
        -------
        """

        # check for no datasets
        assert bidspatient.numdatasets == 0

        # check that resected contacts and original subject id have been added to the participants tsv
        participants_json = bidspatient.loader.load_participants_json
        participants_tsv = bidspatient.loader.load_participants_tsv

        participants_keys = participants_json.keys()
        resect = 'resected_contacts' in participants_keys
        assert resect
        original = 'original_subjectid' in participants_keys
        assert original

        participants_columns = participants_tsv.columns
        resect_col = 'resected_contacts' in participants_columns
        assert resect_col
        original_col = 'original_subjectid' in participants_columns
        assert original_col

        if resect_col:
            for ind, row in participants_tsv.iterrows():
                if (row['participant_id'] == bidspatient.subject_id):
                    resected_contacts = row['resected_contacts']
                    assert resected_contacts ==  " ".join(bidspatient.resected_contacts)
        if original_col:
            for ind, row in participants_tsv.iterrows():
                if (row['participant_id'] == bidspatient.subject_id):
                    original_subjectid = row['original_subjectid']
                    assert original_subjectid == bidspatient.original_subject_id

        # check that the derivate folders have been created




    def test_bidspatient_channelfunc(self, bidspatient):
        """
        Test function for BidsPatient that allows modification of the channels.
        Tests read/write/delete functionality.

        Parameters
        ----------
        bidspatient :

        Returns
        -------

        """
        assert len(bidspatient.wm_channels) >= 0
        assert len(bidspatient.bad_channels) >= 0

        # test working w/ channels
        test_bad_ch = ["a1", "a2", "a3"]
        test_wm_ch = ["b1", "b3"]
        
        bidspatient.add_bad_channels(test_bad_ch)
        bidspatient.add_wm_channels(test_wm_ch)

        assert len(bidspatient.wm_channels) == len(test_wm_ch)
        assert len(bidspatient.bad_channels) == len(test_bad_ch)

        # make sure that bidspatient object isnt loading in any explicit data
        # into the object
        assert sys.getsizeof(bidspatient) < 5 * MB

    def test_bidspatient_errors(self, bidspatient):
        """
        Test error and warnings raised by patient class that should be raised when a user
        does not conform the API.

        Parameters
        ----------
        bidspatient: BidsPatient
            The class object to test

        Returns
        -------
        """
        pass

        
