import os
import tempfile

import pytest

from eegio.base.objects.bids.bids_patient import BidsPatient


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
        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        participants_tsv = participants_tsv.set_index('participant_id')

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
                if (ind == bidspatient.subject_id):
                    resected_contacts = row['resected_contacts']
                    assert resected_contacts == " ".join(bidspatient.resected_contacts)
        if original_col:
            for ind, row in participants_tsv.iterrows():
                if (ind == bidspatient.subject_id):
                    original_subjectid = row['original_subjectid']
                    assert original_subjectid == bidspatient.original_subject_id

        # check that the derivative folders have been created
        data_fpath = os.path.join(bidspatient.datadir, 'sub-' + bidspatient.subject_id, 'ses-' + bidspatient.session_id,
                                  bidspatient.modality)
        prepocessed_fpath = os.path.join(bidspatient.datadir, 'sub-' + bidspatient.subject_id,
                                         'ses-' + bidspatient.session_id, 'preprocessed')
        results_fpath = os.path.join(bidspatient.datadir, 'sub-' + bidspatient.subject_id,
                                     'ses-' + bidspatient.session_id, 'results')

        assert os.path.exists(data_fpath)
        assert os.path.exists(prepocessed_fpath)
        assert os.path.exists(results_fpath)

    def test_bidspatient_participants_modify(self, bidspatient):
        """
        Test function for BidsPatient that allows modification participant files.

        Parameters
        ----------
        bidspatient :

        Returns
        -------

        """
        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        colnames = participants_tsv.columns

        new_field = 'new_field'
        assert new_field not in participants_json.keys()
        assert new_field not in colnames

        # test addition of new field
        bidspatient.add_participants_field(new_field, description='the new field', subject_val='new_val',
                                           default_val='default_val')

        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        colnames = participants_tsv.columns

        assert new_field in participants_json.keys()
        assert new_field in colnames

        # test removal of the field
        bidspatient.remove_participants_field(new_field)
        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        colnames = participants_tsv.columns

        assert new_field not in participants_json.keys()
        assert new_field not in colnames

        bidspatient.add_participants_field(new_field, description='the new field', subject_val='new_val',
                                           default_val='default_val')

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
        subject_id = "test"
        session_id = "seizure"
        managing_user = "EZTrack"
        original_subject_id = 'test_patient_01'
        modality = 'eeg'
        datadir = './bidstest/'
        with tempfile.TemporaryDirectory() as datadir:
            patient = BidsPatient(
                subject_id=subject_id, session_id=session_id, datadir=datadir,
                managing_user=managing_user,
                original_subject_id=original_subject_id, modality=modality
            )

            # assert various user errors that should occur and make sense
