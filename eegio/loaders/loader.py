import os
from typing import Union, List, Tuple

import mne
import numpy as np
import pyedflib

from eegio.base.utils.data_structures_utils import MatReader
from eegio.base.utils.data_structures_utils import loadjsonfile
from eegio.format.baseadapter import BaseAdapter
from eegio.format.scrubber import ChannelScrub, EventScrub
from eegio.loaders.baseloader import BaseLoader


class Loader(BaseLoader):
    def __init__(self, fname, metadata: dict = {}):
        super(Loader, self).__init__(fname=fname)

        self.metadata = metadata

    def load_file(self, filepath: os.PathLike):
        pass

    def load_adapter(self, adapter: BaseAdapter):
        self.adapter = adapter

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

    def read_npzjson(self, jsonfpath: os.PathLike, npzfpath: os.PathLike = None):
        """
        Reads a numpy stored as npz+json file combination.

        :param jsonfpath:
        :type jsonfpath:
        :param npzfpath:
        :type npzfpath:
        :return:
        :rtype:
        """
        filedir = os.path.dirname(jsonfpath)
        # load in json file
        metadata = loadjsonfile(jsonfpath)

        if npzfpath == None:
            npzfilename = metadata["resultfilename"]
            npzfpath = os.path.join(filedir, npzfilename)

        datastruct = np.load(npzfpath)
        return datastruct, metadata

    def read_npyjson(self, jsonfpath: os.PathLike, npyfpath: os.PathLike = None):
        """
        Reads a numpy stored as npy+json file combination.

        :param jsonfpath:
        :type jsonfpath:
        :param npyfpath:
        :type npyfpath:
        :return:
        :rtype:
        """
        filedir = os.path.dirname(jsonfpath)
        # load in json file
        metadata = loadjsonfile(jsonfpath)

        if npyfpath == None:
            npyfilename = metadata["resultfilename"]
            npyfpath = os.path.join(filedir, npyfilename)

        arr = np.load(npyfpath)
        return arr, metadata

    def read_edf(
            self,
            fname,
            backend: str = "mne",
            montage=None,
            eog: Union[List, Tuple] = None,
            misc: Union[List, Tuple] = None,
            linefreq: float = 60,
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
            # use mne to read the raw edf, events and the info data struct
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
        # raw.info["chs"] = ChannelScrub.label_channel_types(raw.info["chs"])

        # add line frequency
        raw.info["line_freq"] = linefreq

        return raw, annotations

    def read_fif(self, fname, linefreq: float = 60):
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
        # raw.info["chs"] = ChannelScrub.label_channel_types(raw.info["chs"])

        annotations = raw.annotations

        return raw, annotations

    def read_mat(self, fname, linefreq: float = 60):
        if linefreq not in [50, 60]:
            raise ValueError(
                "Line frequency should be set to a valid number! "
                f"USA is 60 Hz, and EU is 50 Hz. You passed: {linefreq}."
            )

        reader = MatReader()
        datastruct = reader.loadmat(fname)

        raise RuntimeWarning("Need to explicitly set line frequency in code.")
        # return datastruct
