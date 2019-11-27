import os
import tempfile
import mne
import pytest

from eegio.base.objects.bids.bids_patient import BidsPatient


@pytest.mark.usefixture("bidspatient")
class TestBidsPatient:
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
        # assert bidspatient.numdatasets == 0

        # check the participants tsv
        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        participants_tsv = participants_tsv[participants_tsv["participant_id"] == bidspatient.subject_id].to_dict()
        participant_metadata = bidspatient.load_subject_metadata()
        assert all(participant_metadata[key] == participants_tsv[key] for key in participants_tsv)
        assert participant_metadata["field_descriptions"] == participants_json

        '''
        Load runs from BidsPatient
        '''
        test_session = "seizure"
        test_run = "01"
        raw = bidspatient.load_run(session_id=test_session,
                             run_id=test_run)
        assert isinstance(raw, mne.io.BaseRaw)

        rawlist = bidspatient.load_session(session_id=test_session)
        assert all(isinstance(x, mne.io.BaseRaw) for x in rawlist)

        '''
        Check that all the metadata is done correctly
        '''

    @pytest.mark.skip()
    def test_bidspatient_addfields(self, bidspatient):
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

        # check the participants tsv
        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        participants_tsv = participants_tsv.set_index('participant_id')

        participants_keys = participants_json.keys()
        resect = 'resected_contacts' in participants_keys
        assert not resect
        original = 'original_subjectid' in participants_keys
        assert not original

        participants_columns = participants_tsv.columns
        resect_col = 'resected_contacts' in participants_columns
        assert not resect_col
        original_col = 'original_subjectid' in participants_columns
        assert not original_col

        bidspatient.add_resected_contacts(['a1','a2'])
        bidspatient.add_original_subject_id('test_0001')

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

    @pytest.mark.skip()
    def test_bidspatient_participants_modify(self, bidspatient):
        """
        Test function for BidsPatient that allows modification participant files.

        Parameters
        ----------
        bidspatient :

        Returns
        -------

        """
        ''' 
        Tests, adding a new subject field
        '''
        subject_metadata = bidspatient.load_subject_metadata()
        new_field = 'new_field'
        assert new_field not in subject_metadata.keys()

        # test addition of new field
        subject_metadata = bidspatient.add_participants_field(new_field,
                                           description='the new field',
                                           subject_val='new_val',
                                           default_val='default_val')
        assert new_field in subject_metadata.keys()

        # test to make sure it was actually loaded in properly
        subject_metadata = bidspatient.load_subject_metadata()
        assert new_field in subject_metadata.keys()

        '''
        Tests, deleting an existing subject field
        '''
        # test removal of the field
        bidspatient.remove_participants_field(new_field)
        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        colnames = participants_tsv.columns

        assert new_field not in participants_json.keys()
        assert new_field not in colnames

        bidspatient.add_participants_field(new_field, description='the new field', subject_val='new_val',
                                           default_val='default_val')

        '''
        Tests, modifying an existing subject field
        '''

    @pytest.mark.skip()
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
        new_field = 'new_field'
        # error should be raised when trying to remove a field that does not exist
        with pytest.raises(LookupError):
            bidspatient.add_participants_field(new_field,
                                               description='the new field', subject_val='new_val',
                                               default_val='default_val')
            bidspatient.remove_participants_field(new_field)
            bidspatient.remove_participants_field(new_field)

        fake_fpaths = ['Path/That/Does/Not/Exist.edf']
        # errors should be raised when passing in files that do not exist.
        with pytest.raises(OSError):
            bidspatient.add_scans(fake_fpaths)

        with pytest.raises(OSError):
            bidspatient.preprocess_edf_files(fake_fpaths)

        # error should be raised when attempting to modify the participants tsv with a column name that DNE
        with pytest.raises(ValueError):
            bidspatient.modify_participants_file(column_id='nonexisting column', new_value='new_value')

    def test_bidspatient_catastrophic(self, bidspatient):
        pass