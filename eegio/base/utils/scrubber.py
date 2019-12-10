import re
from typing import List, Dict

import mne
import numpy as np

from eegio.base.objects.electrodes.elecs import Contacts
from eegio.base.config import NON_EEG_MARKERS, BAD_MARKERS


class ChannelScrub:
    @classmethod
    def channel_text_scrub(cls, raw: mne.io.BaseRaw):
        """
        Cleans and formats the channel text inside a MNE-Raw data structure.

        TODO: ACCOUNT FOR - BEING IN BIPOLAR MONTAGE CHANNEL LABELS.

        Parameters
        ----------
        raw : MNE-raw data structure

        Returns
        -------

        """

        def _reformatchanlabel(label):
            """
            Helper function to process a single channel label to make sure it is:

            - upper case
            - removed unnecessary strings (POL, eeg, -ref)
            - removed empty spaces

            :param label: (str) a contact label that may have extra chars, or be improperly cased
            :return: label (str) the reformatted string that is uppercase and w/o spaces
            """
            # hard coded replacement rules
            # label = str(label).replace("POL ", "").upper()
            label = str(label).replace("POL", "").upper()
            label = label.replace("EEG", "").replace("-REF", "")

            # replace "Grid" with 'G' label
            label = label.replace("GRID", "G")
            # for BIDS format, you cannot have blank channel name
            if label == "":
                label = "N/A"
            return label

        # apply channel scrubbing
        raw.rename_channels(lambda x: x.upper())
        raw.rename_channels(lambda x: x.strip("."))  # remove dots from channel names
        raw.rename_channels(lambda x: x.strip("-"))  # remove dashes from channel names
        raw.rename_channels(lambda x: x.replace(" ", ""))
        raw.rename_channels(
            lambda x: x.replace("’", "'")
        )  # remove dashes from channel names
        raw.rename_channels(
            lambda x: x.replace("`", "'")
        )  # remove dashes from channel names
        raw.rename_channels(lambda x: _reformatchanlabel(x))

        return raw

    @classmethod
    def look_for_bad_channels(self, ch_names, bad_markers: List[str] = BAD_MARKERS):
        """
        Looks for hardcoding of what are "bad channels"

        Parameters
        ----------
        ch_names : (list) a list of str channel labels
        bad_markers : (list) of string labels

        Returns
        -------

        """
        orig_chdict = {ch.upper(): ch for ch in ch_names}

        ch_names = [c.upper() for c in ch_names]

        # initialize a list to store channel label strings
        bad_channels = []

        # look for channels without letter
        bad_channels.extend([ch for ch in ch_names if not re.search("[a-zA-Z]", ch)])
        # look for channels that only have letters - turn off for NIH pt17
        letter_chans = [ch for ch in ch_names if re.search("[a-zA-Z]", ch)]
        bad_channels.extend([ch for ch in letter_chans if not re.search("[0-9]", ch)])

        if "$" in bad_markers:
            # look for channels with '$'
            bad_channels.extend([ch for ch in ch_names if re.search("[$]", ch)])
        if "FZ" in bad_markers:
            badname = "FZ"
            bad_channels.extend([ch for ch in ch_names if ch == badname])
        if "GZ" in bad_markers:
            badname = "GZ"
            bad_channels.extend([ch for ch in ch_names if ch == badname])
        if "DC" in bad_markers:
            badname = "DC"
            bad_channels.extend([ch for ch in ch_names if badname in ch])
        if "STI" in bad_markers:
            badname = "STI"
            bad_channels.extend([ch for ch in ch_names if badname in ch])

        # extract non eeg channels based on some rules we set
        non_eeg_channels = [
            chan for chan in ch_names if any(x in chan for x in NON_EEG_MARKERS)
        ]
        # get rid of these channels == 'e'
        non_eeg_channels.extend([ch for ch in ch_names if ch == "E"])
        bad_channels.extend(non_eeg_channels)

        bad_channels = [orig_chdict[ch] for ch in bad_channels]
        return bad_channels

    @classmethod
    def label_channel_types(cls, labels: List[str]):
        """
        Load in the channel types and try to infer them.
        The possibilities are: EEG, STIM, EOG, EKG, Misc. that are from MNE-Python.

        We map these to:
        1. bad-non: bad or non-eeg channels
        2. grid: grid channels (1-k*8 contacts)
        3. strip: strip channels (1-6, or 1-8 contacts)
        4. seeg: depth channels inserted (1-8 up to 1-16)

        Parameters
        ----------
        labels :

        Returns
        -------

        """

        def remove_letters(s):
            no_digits = []
            # Iterate through the string, adding non-numbers to the no_digits list
            for i in s:
                if i.isdigit():
                    no_digits.append(i)

            # Now join all elements of the list with '',
            # which puts all of the characters together.
            result = "".join(no_digits)
            return result

        contacts = Contacts(contacts_list=labels, require_matching=False)

        # create hash dictionary to store label of each channel
        channeltypes = {}
        for chanlabel in contacts.chanlabels:
            eleclabel = chanlabel
            # get electrode label for this channel
            # eleclabel = contacts.get_elec(chanlabel)
            # get rest of electrode labels
            # elec_contacts_nums = [
            #     int(remove_letters(labels[ind]))
            #     for ind in contacts.electrodes[eleclabel]
            # ]

            # if elec_contacts_nums == []:
            #     channeltypes[chanlabel] = "BAD"
            if any(x in eleclabel for x in ["EKG", "ECG"]):
                channeltypes[chanlabel] = "ecg"
            elif "EOG" in eleclabel:
                channeltypes[chanlabel] = "eog"
            elif "EMG" in eleclabel:
                channeltypes[chanlabel] = "emg"
            elif "STIM" in eleclabel:
                channeltypes[chanlabel] = "stim"
            # elif eleclabel == "G":
            #     channeltypes[chanlabel] = "GRID"
            # elif max(elec_contacts_nums) <= 6:
            #     channeltypes[chanlabel] = "STRIP"
            # elif max(elec_contacts_nums) > 6 and max(elec_contacts_nums) < 20:
            #     channeltypes[chanlabel] = "SEEG"
            # else:
            #     channeltypes[chanlabel] = "EEG"

        return channeltypes


class EventScrub:
    @classmethod
    def find_seizure_onset(
        cls,
        event_onsets: List[int],
        event_durations: List[float],
        event_keys: List[int],
        event_ids: Dict,
        offset_time: float = None,
        multiple_sz: bool = False,
        onset_marker_name: str = "",
    ):
        """
        Eventscrubber to determine where seizure onset is and return the marker (in seconds)
        after the recording start. E.g. recording starts at 0, and seizure occurs at 45 seconds.
        If sampling rate was 1000 Hz, then seizure index is 45000.

        :param event_onsets:
        :param event_durations:
        :param event_keys:
        :param event_ids:
        :param offset_time:
        :param multiple_sz:
        :param onset_marker_name:
        :return:
        """
        # onset markers
        onsetmarks = ["onset", "crise", "cgtc", "sz", "absence"]

        # if an explicit onset marker name is passed
        if onset_marker_name:
            # find name where it occurs
            eventid = event_ids[onset_marker_name]
            idx = np.where(event_keys == eventid)[0][0]
            onset_secs = event_onsets[idx].astype(float)
            return onset_secs

        # if not, then parse through possible markers
        for name, eventid in event_ids.items():
            name = ",".join(name.upper().split(" "))

            # search for onset markers
            if any(re.search(r"\b{}\b".format(x), name) for x in onsetmarks):
                # find index where onset marker name occurs and get the corresponding time
                idx = np.where(event_keys == eventid)[0][0]
                onset_secs = event_onsets[idx].astype(float)

                # if event durations is > 0
                if event_durations[idx] > 0:
                    onset_secs = onset_secs + (event_durations[idx] / 2)
                    raise RuntimeWarning(
                        "Event durations is > 0 for a seizure marker?"
                        " Could be an error."
                    )

                # check if we passed in offset time, onset can't be after offset
                if offset_time:
                    if offset_time < onset_secs:
                        continue

                if not multiple_sz:
                    return onset_secs

        return None

    @classmethod
    def find_seizure_offset(
        cls,
        event_onsets: List[int],
        event_durations: List[float],
        event_keys: List[int],
        event_ids: Dict,
        onset_time: float = None,
        multiple_sz: bool = False,
        offset_marker_name: str = "",
    ):
        """
        Eventscrubber to determine where seizure offset is and return the marker (in seconds)
        after the recording start. E.g. recording starts at 0, and seizure offset occurs at 45 seconds.
        If sampling rate was 1000 Hz, then seizure offset index is 45000. It is a good idea to pass in the
        onset marker, to make sure offset marker occurs AFTERwards.

        :param event_onsets:
        :param event_durations:
        :param event_keys:
        :param event_ids:
        :param onset_time:
        :param multiple_sz:
        :param offset_marker_name:
        :return:
        """
        offsetmarks = ["offset", "fin", "end", "over"]

        # if an explicit onset marker name is passed
        if offset_marker_name:
            # find name where it occurs
            eventid = event_ids[offset_marker_name]
            idx = np.where(event_keys == eventid)[0][0]
            offset_secs = event_onsets[idx].astype(float)
            return offset_secs

        # if not, then parse through possible markers
        for name, eventid in event_ids.items():
            name = ",".join(name.upper().split(" "))

            # search for offset markers
            if any(re.search(r"\b{}\b".format(x), name) for x in offsetmarks):
                # find index where onset marker name occurs and get the corresponding time
                idx = np.where(event_keys == eventid)[0][0]
                offset_secs = event_onsets[idx].astype(float)

                # if event durations is > 0
                if event_durations[idx] > 0:
                    offset_secs = offset_secs + (event_durations[idx] / 2)
                    raise RuntimeWarning(
                        "Event durations is > 0 for a seizure marker?"
                        " Could be an error."
                    )

                # check if we passed in onset time, onset can't be after offset
                if onset_time:
                    if onset_time > offset_secs:
                        continue

                # check if multiple seizures should be looked for
                if not multiple_sz:
                    return offset_secs
                else:
                    raise RuntimeError("Can't handle multiple seizures in file yet.")
        return None
