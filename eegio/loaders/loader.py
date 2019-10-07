import os
from typing import Union, List, Tuple, Dict

import mne
import numpy as np
import deprecated
import pyedflib

from eegio.base.objects import EEGTimeSeries, Contacts
from eegio.base.utils.data_structures_utils import MatReader
from eegio.format.baseadapter import BaseAdapter
from eegio.format.scrubber import ChannelScrub, EventScrub
from eegio.loaders.baseloader import BaseLoader


class Loader(BaseLoader):
    def __init__(self, fname, metadata: Dict = None):
        super(Loader, self).__init__(fname=fname)

        if metadata is None:
            metadata = {}
        self.update_metadata(**metadata)

    def load_file(self, filepath: Union[str, os.PathLike]):
        filepath = str(filepath)
        if filepath.endswith(".fif"):
            res = self.read_fif(filepath)
        elif filepath.endswith(".mat"):
            res = self.read_mat(filepath)
        elif filepath.endswith(".edf"):
            res = self.read_edf(filepath)
        else:
            raise OSError("Can't use load_file for this file extension {filepath} yet.")

        return res

    def load_adapter(self, adapter: BaseAdapter):
        self.adapter = adapter

    def _wrap_raw_in_obj(
        self, raw_mne: mne.io.BaseRaw, modality: str, annotations: List
    ):
        """
        Helps wrap a MNE.io.Raw object into a light-weight wrapper object representing
        the EEG time series, adds additional metadata.

        :param raw_mne:
        :type raw_mne:
        :param modality:
        :type modality:
        :return:
        :rtype:
        """
        # load data into RAM
        raw_mne.load_data()

        # scrub channels
        raw_mne = ChannelScrub.channel_text_scrub(raw_mne)
        chlabels = raw_mne.ch_names
        # look for bad channels
        badchs = ChannelScrub.look_for_bad_channels(chlabels)
        # goodinds = [i for i, ch in enumerate(chlabels) if ch not in badchs]
        # drop the bad channels
        raw_mne.drop_channels(badchs)
        chlabels = raw_mne.ch_names
        # label channels
        labels_of_ch = ChannelScrub.label_channel_types(chlabels)

        # load in the cleaned ch labels
        contacts = Contacts(chlabels, require_matching=False)
        samplerate = raw_mne.info["sfreq"]
        rawdata, times = raw_mne.get_data(return_times=True)

        # create EEG TS object
        eegts = EEGTimeSeries(rawdata, times, contacts, samplerate, modality)
        eegts.update_metadata(raw_annotations=annotations)
        return eegts

    def _wrap_mat_in_obj(self, datastruct, modality: str, annotations: List):
        chlabels = datastruct["chlabels"]
        rawdata = datastruct["data"]
        samplerate = datastruct["sfreq"]

        # look for bad channels
        badchs = ChannelScrub.look_for_bad_channels(chlabels)
        goodinds = [i for i, ch in enumerate(chlabels) if ch not in badchs]
        # drop the bad channels
        rawdata = rawdata[goodinds, ...]
        chlabels = chlabels[goodinds]
        times = np.arange(0, rawdata.shape[1])

        # label channels
        labels_of_ch = ChannelScrub.label_channel_types(chlabels)

        # load in the cleaned ch labels
        contacts = Contacts(chlabels, require_matching=False)

        # create EEG TS object
        eegts = EEGTimeSeries(
            rawdata, times, contacts, samplerate, modality, metadata=self.metadata_dict
        )
        eegts.update_metadata(raw_annotations=annotations)
        return eegts

    def read_NK(self, fname):
        """
        Function to read from a Nihon-Kohden based EEG system file.

        :param fname:
        :type fname:
        :return:
        :rtype:
        """
        pass

    def read_Natus(self, fname: os.PathLike):
        """
        Function to read from a Natus based EEG system file.
        :param fname:
        :type fname:
        :return:
        :rtype:
        """
        pass

    def read_edf(
        self,
        fname,
        backend: str = "mne",
        montage=None,
        eog: Union[List, Tuple] = None,
        misc: Union[List, Tuple] = None,
        linefreq: float = 60,
        modality: str = "eeg",
        return_mne: bool = False,
    ):
        """
        Function to read in edf file either using MNE, or PyEDFLib. Recommended to use
        MNE.

        :param fname:
        :type fname:
        :param backend:
        :type backend:
        :param montage:
        :type montage:
        :param eog:
        :type eog:
        :param misc:
        :type misc:
        :param linefreq:
        :type linefreq:
        :return:
        :rtype:
        """
        if linefreq not in [50, 60]:
            raise ValueError(
                "Line frequency should be set to a valid number! "
                f"USA is 60 Hz, and EU is 50 Hz. You passed: {linefreq}."
            )

        # HARD CODED excluded contacts. Don't read in blanks and dashes
        excluded_contacts = ["-", ""]

        # read edf file w/ certain backend
        if backend == "mne":
            # use mne to read the raw edf, events and the metadata data struct
            raw = mne.io.read_raw_edf(
                fname,
                preload=True,
                verbose="ERROR",
                exclude=excluded_contacts,
                montage=montage,
                eog=eog,
                misc=misc,
            )
            samplerate = raw.info["sfreq"]

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

            if any(isinstance(x, float) for x in eventonsets):
                raise RuntimeError(
                    "We are assuming event onsets are in their index form."
                    "To convert to seconds, need to divide by sample rate."
                )
            eventonsets = np.divide(eventonsets, samplerate)

            # extract seizure onsets/offsets
            seizure_onset_sec = EventScrub().find_seizure_onset(
                eventonsets, eventdurations, eventkeys, event_ids
            )
            seizure_offset_sec = EventScrub().find_seizure_offset(
                eventonsets,
                eventdurations,
                eventkeys,
                event_ids,
                onset_time=seizure_onset_sec,
            )

            onsetind = (
                int(np.multiply(seizure_onset_sec, samplerate))
                if seizure_onset_sec != None
                else None
            )
            offsetind = (
                int(np.multiply(seizure_offset_sec, samplerate))
                if seizure_offset_sec != None
                else None
            )
            self.update_metadata(onset=seizure_onset_sec, onsetind=onsetind)
            self.update_metadata(offset=seizure_offset_sec, offsetind=offsetind)

        elif backend == "pyedflib":
            # raw = pyedflib.
            raw = pyedflib.EdfReader(fname)
            n = raw.signals_in_file
            signal_labels = raw.getSignalLabels()
            sigbufs = np.zeros((n, raw.getNSamples()[0]))
            for i in np.arange(n):
                sigbufs[i, :] = raw.readSignal(i)

            # get the annotations
            annotations = raw.readAnnotations()

        else:
            raise AttributeError(
                "backend supported for reading edf files are: mne and pyedflib. "
                f"You passed in {backend}. Please choose an appropriate version."
            )

        # scrub channel labels:
        raw = ChannelScrub.channel_text_scrub(raw)
        # raw.metadata["chs"] = ChannelScrub.label_channel_types(raw.metadata["chs"])

        # add line frequency
        raw.info["line_freq"] = linefreq

        if return_mne:
            return raw, annotations
        else:
            eegts = self._wrap_raw_in_obj(raw, modality, annotations)
            return eegts

    def read_fif(
        self,
        fname,
        linefreq: float = 60,
        modality: str = "eeg",
        return_mne: bool = False,
    ):
        """
        Function to read in a .fif type file using MNE.

        :param fname:
        :type fname:
        :param linefreq:
        :type linefreq:
        :return:
        :rtype:
        """
        if linefreq not in [50, 60]:
            raise ValueError(
                "Line frequency should be set to a valid number! "
                f"USA is 60 Hz, and EU is 50 Hz. You passed: {linefreq}."
            )

        # extract raw object
        if fname.endswith(".fif"):
            raw = mne.io.read_raw_fif(fname, preload=False, verbose=False)
        else:
            raise ValueError(
                "All files read in with eegio need to be preformatted into fif first!"
            )

        # add line frequency
        if raw.info["line_freq"] == None:
            raw.info["line_freq"] = linefreq

        # scrub channel labels:
        raw = ChannelScrub.channel_text_scrub(raw)
        annotations = raw.annotations

        if return_mne:
            return raw, annotations
        else:
            eegts = self._wrap_raw_in_obj(raw, modality, annotations)
            return eegts

    @deprecated.deprecated(reason="mat loading may be unnecessary for now.")
    def read_mat(
        self,
        fname,
        linefreq: float = 60,
        modality: str = "eeg",
        return_struct: bool = False,
    ):
        """
        Function to read in a .mat type file and convert if necessary into an EEGTimeSeries.

        :param fname:
        :type fname:
        :param linefreq:
        :type linefreq:
        :param modality:
        :type modality:
        :param return_struct:
        :type return_struct:
        :return:
        :rtype:
        """
        if linefreq not in [50, 60]:
            raise ValueError(
                "Line frequency should be set to a valid number! "
                f"USA is 60 Hz, and EU is 50 Hz. You passed: {linefreq}."
            )
        reader = MatReader()
        datastruct = reader.loadmat(fname)

        errmessage = []
        if "chlabels" not in datastruct.keys():
            errstr = (
                f"'chlabels' needs to be part of the datastruct in your .mat file. "
                f"If not, then you need to add it to use read_mat."
            )
            errmessage.append(errstr)
        if "data" not in datastruct.keys():
            errstr = (
                f"'data' needs to be part of the datastruct in your .mat file. "
                f"If not, then you need to add it to use read_mat."
            )
            errmessage.append(errstr)
        if "sfreq" not in datastruct.keys():
            errstr = (
                f"'sfreq' needs to be part of the datastruct in your .mat file. "
                f"If not, then you need to add it to use read_mat."
            )
            errmessage.append(errstr)
        if errmessage:
            raise RuntimeError(errmessage)

        if "annotations" in datastruct.keys():
            annotations = datastruct["annotations"]
        else:
            annotations = []

        if return_struct:
            return datastruct, annotations
        else:
            eegts = self._wrap_mat_in_obj(datastruct, modality, annotations)
            return eegts
