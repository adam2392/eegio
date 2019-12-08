import mne
import pytest

from eegio.base.utils.scrubber import ChannelScrub


@pytest.mark.usefixtures("bidsrun_scalp")
class TestBidsRun:
    def testbidsrun_basicfunc(self, bidsrun_scalp):
        """
        Tests the main BidsRun functionality. A BidsRun should be able to:

        1. Load data: return MNE objects along with metadata
        2. modify the sidecar json:
        3. append channel information
        4. modify channel information

        Parameters
        ----------
        bidspatient : BidsPatient

        Returns
        -------

        """
        # do some sanity checks. If USA data, make sure linefreq 60, else 50
        assert bidsrun_scalp.linefreq in [50, 60]
        assert bidsrun_scalp.sfreq >= 200

        # get the channel labels
        chs = bidsrun_scalp.chs

        # determine all the electrode types
        chtypes = bidsrun_scalp.get_channel_types()

        # determine all the channel status
        chstatus = bidsrun_scalp.get_channels_status()

        # get good and bad indices based on status
        badinds = [i for i, ch in enumerate(chs) if chstatus["status"][ch] == "bad"]
        goodinds = [i for i in range(len(chs)) if i not in badinds]

        # loaded in the scalp EEG data
        raw = bidsrun_scalp.load_data()
        assert isinstance(raw, mne.io.base.BaseRaw)

    def testbidsrun_modifyfunc(self, bidsrun_scalp):
        """
        Tests the BidsRun functionality to modify metadata in the context of a sidecar json and channels.tsv
        file. It should be able to:

        1. Modify BIDS columns for each channel name
        2. Modify sidecar json metadata related to a bidsrun.

        Parameters
        ----------
        bidsrun_scalp :

        Returns
        -------

        """
        # get the channel labels
        chs = bidsrun_scalp.chs

        """
        First test adding additional metadata using the sidecarjson.
        """
        # test field not default written into sidecar
        institution_name = "National institute of Health"
        bidsrun_scalp.delete_sidecar_field(key="InstitutionName")
        run_metadata = bidsrun_scalp.modify_sidecar_field(
            key="InstitutionName", value=institution_name, overwrite=False
        )
        assert bidsrun_scalp.get_run_metadata()["InstitutionName"] == institution_name
        # this should have been written

        # test for capitalization specificity
        manufacturer = "nk"
        run_metadata = bidsrun_scalp.modify_sidecar_field(
            key="manufacturer", value=manufacturer, overwrite=True
        )
        assert bidsrun_scalp.get_run_metadata()["Manufacturer"] == manufacturer
        # Keynames should always start with capital letter when passed in
        assert all(
            x in bidsrun_scalp.get_run_metadata().keys()
            for x in ["Manufacturer", "InstitutionName"]
        )

        # test for field that needs to be overwritten
        institution_name = "National institute of Health_v2"
        run_metadata = bidsrun_scalp.modify_sidecar_field(
            key="InstitutionName", value=institution_name, overwrite=True
        )
        # this should have been rewritten
        assert bidsrun_scalp.get_run_metadata()["InstitutionName"] == institution_name

        """
        Second test adding additional metadata using the channels tsv file.
        """
        # determine all the electrode types
        chtypes = bidsrun_scalp.get_channel_types()
        # determine all the channel status
        chstatus = bidsrun_scalp.get_channels_status()

        # overwrite for all channels some column, by calling function
        column_id = "low_cutoff"
        value = "0.0"
        for channel_id in chs:
            bidsrun_scalp.modify_channel_info(
                channel_id=channel_id, column_id=column_id, value=value
            )

        # append new channel information
        column_id = "resected"
        value = True
        for channel_id in chs:
            bidsrun_scalp.append_channel_info(
                column_id=column_id, channel_id=channel_id, value=value
            )

    def testbidsrun_errors(self, bidsrun_scalp):
        # get the channel labels
        chs = bidsrun_scalp.chs

        # assert errors that should arise when run tries to write certain data types
        with pytest.raises(RuntimeError):
            institution_name = "National institute of Health"
            run_metadata = bidsrun_scalp.modify_sidecar_field(
                key="InstitutionName", value=institution_name
            )
            institution_name = "National institute of Health_v2"
            run_metadata = bidsrun_scalp.modify_sidecar_field(
                key="InstitutionName", value=institution_name
            )

        # error should be raised when both dictionary mapping and channel w/ default value is passed in
        with pytest.raises(
            TypeError,
            match="Passed in both value and channel dictionary. " "Only pass in one!",
        ):
            bidsrun_scalp.append_channel_info(
                column_id="test",
                channel_id=chs[0],
                value="test",
                channel_dict={ch: "test" for ch in chs},
            )

        # error should be raised when attempting to modify a channel that does not exist
        with pytest.raises(LookupError):
            bidsrun_scalp.modify_channel_info(
                column_id="low_cutoff", channel_id="Nonexisting channel", value="2.0"
            )

        # error should be raised when attempting to modify a column that does not exist
        with pytest.raises(ValueError):
            bidsrun_scalp.modify_channel_info(
                column_id="Nonexisting column", channel_id="EEG C4-Ref", value=True
            )

        # run scrubber through "good channels" and make sure internal functionality didn't mess things up
        # get the channel labels
        chs = bidsrun_scalp.chs
        # determine all the channel status
        chstatus = bidsrun_scalp.get_channels_status()
        # get good and bad indices based on status
        badinds = [i for i, ch in enumerate(chs) if chstatus["status"][ch] == "bad"]
        goodinds = [i for i in range(len(chs)) if i not in badinds]
        badchs = ChannelScrub.look_for_bad_channels(chs[goodinds])
        with pytest.raises(Exception):
            assert badchs == []

    def testbidsrun_catastrophic(self, bidsrun_scalp):
        """
        A test for descriptive errors in catastrophic error cases. These include, but are not limited to:

        1.

        TODO:
        - complete

        Parameters
        ----------
        bidsrun_scalp :

        Returns
        -------

        """
        pass
