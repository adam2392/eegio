import mne
import pytest

from eegio.loaders.bids import BidsPatient


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
        # check the participants tsv
        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        participants_tsv = participants_tsv[
            participants_tsv["participant_id"] == bidspatient.subject_id
        ].to_dict()
        participant_metadata = bidspatient.load_subject_metadata()
        # assert all(participant_metadata[key] == participants_tsv[key] for key in participants_tsv.keys())
        assert participant_metadata["field_descriptions"] == participants_json

        print("Testing: ", bidspatient)

        """
        Load runs from BidsPatient
        """
        test_session = "seizure"
        test_run = "01"
        raw = bidspatient.load_run(session_id=test_session, run_id=test_run)
        assert isinstance(raw, mne.io.BaseRaw)

        rawlist = bidspatient.load_session(session_id=test_session)
        assert all(isinstance(x, mne.io.BaseRaw) for x in rawlist)

        """
        Check that all the metadata is done correctly
        """
        subj_sessions = bidspatient.get_subject_sessions()
        subj_kinds = bidspatient.get_subject_kinds()
        subj_runs = bidspatient.get_subject_runs()
        assert len(subj_sessions) == 1
        assert len(subj_kinds) == 1
        assert len(subj_runs) == 2
        assert len(bidspatient.edf_fpaths) == 2

    # @pytest.mark.skip()
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
        """
        Check that we can add fields to the run level for all participants.
        
        E.g. adding resected contacts, etc.
        """

        # add list to participants description
        resected_contacts = ["A1", "A2", "A3", "A4"]
        resected_str = " ".join(resected_contacts)
        bidspatient.add_participants_field(
            column_id="resected_contacts",
            description="The list of contacts corresponding to resected regions",
            subject_val=resected_str,
            default_val="n/a",
        )
        bidspatient.remove_participants_field("resected_contacts")

    # @pytest.mark.skip()
    def test_bidspatient_participants_modify(self, bidspatient):
        """
        Test function for BidsPatient that allows modification participant files.

        Parameters
        ----------
        bidspatient :

        Returns
        -------

        """
        """
        Tests, adding a new subject field. Check that we can add fields to the participants level
        
        """
        subject_metadata = bidspatient.load_subject_metadata()
        new_field = "new_field"
        assert new_field not in subject_metadata.keys()

        # test addition of new field
        bidspatient.add_participants_field(
            new_field,
            description="the new field",
            subject_val="new_val",
            default_val="default_val",
        )
        subject_metadata = bidspatient.metadata
        print(subject_metadata)
        assert new_field in subject_metadata.keys()

        # test to make sure it was actually loaded in properly
        subject_metadata = bidspatient.load_subject_metadata()
        assert new_field in subject_metadata.keys()

        """
        Tests, deleting an existing subject field
        """
        # test removal of the field
        bidspatient.remove_participants_field(new_field)
        participants_json = bidspatient.loader.load_participants_json()
        participants_tsv = bidspatient.loader.load_participants_tsv()
        colnames = participants_tsv.columns
        assert new_field not in participants_json.keys()
        assert new_field not in colnames

        """
        Tests, modifying an existing subject field
        """
        bidspatient.modify_participants_file(column_id="age", new_value=40)
        subject_metadata = bidspatient.load_subject_metadata()
        print(subject_metadata)
        assert subject_metadata["age"] == 40

    # @pytest.mark.skip()
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
        new_field = "new_field"
        # error should be raised when trying to remove a field that does not exist
        with pytest.raises(LookupError):
            bidspatient.add_participants_field(
                column_id=new_field,
                description="the new field",
                subject_val="new_val",
                default_val="default_val",
            )
            bidspatient.remove_participants_field(new_field)
            bidspatient.remove_participants_field(new_field)

        # error should be raised when attempting to modify the participants tsv with a column name that DNE
        with pytest.raises(ValueError):
            bidspatient.modify_participants_file(
                column_id="nonexisting column", new_value="new_value"
            )

    def test_bidspatient_catastrophic(self, bidspatient):
        pass
