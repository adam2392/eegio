import pytest
import mne
from eegio.base.utils.scrubber import ChannelScrub, EventScrub


@pytest.mark.usefixture("edf_fpath")
class TestContacts:
    def test_channelscrubber(self):
        """
        Test error and warnings raised by Contacts class.

        :param contacts: (Contacts)
        :return: None
        """
        chlabels = ["$", "", "-", "A1", "a1", "a2"]
        scrubber = ChannelScrub()

        badchs = scrubber.look_for_bad_channels(chlabels)
        chlabeled_dict = scrubber.label_channel_types(chlabels)

    def test_eventscrubber(self, edf_fpath):
        # use mne to read the raw edf, events and the metadata data struct
        raw = mne.io.read_raw_edf(edf_fpath, preload=True, verbose="ERROR")
        # get the annotations
        annotations = mne.events_from_annotations(
            raw, use_rounding=True, chunk_duration=None
        )
        event_times, event_ids = annotations

        # split event ids into their markernames and key-identifier
        eventnames = event_ids.keys()

        # extract the 3 columns from event times
        # onset relative to start of recording
        eventonsets = event_times[:, 0]
        eventdurations = event_times[:, 1]  # duration of event
        eventkeys = event_times[:, 2]  # key-identifier

        # test_eventonsets = [20, 30, 40]
        # test_eventdurations = [0, 0, 0]
        # test_eventkeys = ["sz onset", "sz offset", "done"]
        # test_eventids = [0, 1, 2]

        multiplesz = False
        # pass case for finding onset/offset, marker
        eventscrubber = EventScrub()

        print(zip(eventonsets, eventdurations, eventkeys))

        szonset = eventscrubber.find_seizure_onset(
            eventonsets, eventdurations, eventkeys, event_ids, multiple_sz=multiplesz
        )
        szoffset = eventscrubber.find_seizure_offset(
            eventonsets, eventdurations, eventkeys, event_ids, multiple_sz=multiplesz
        )
        # assert szonset
        # assert szoffset
