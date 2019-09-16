import os

import mne
import numpy as np
from natsort import natsorted

from eegio.loaders.base.baseio import BaseIO

try:
    to_unicode = unicode
except NameError:
    to_unicode = str


class BaseRawLoader(BaseIO):
    """
    Base class for any raw data loader (i.e. scalp EEG, SEEG, ECoG) eeg time series data.

    Raw data should have accompanying metadata with it that is easily parsed as .json, .mat, .npz, or .hdf, which
    will be loaded into the class objects as properties.

    Also allows for filtering functionality here and channel masking.

    Attributes
    ----------
    root_dir : os.PathLike
        The root directory that datasets will be located.

    Notes
    -----
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load_metadata(self, metadata):
        self.metadata = metadata
        # extract type information and other channel information
        if "type" in metadata.keys():
            self.type = metadata["type"]
        else:
            self.type = None

        # extract patient notes
        if "note" in metadata.keys():
            self.note = metadata["note"]
        else:
            self.note = None

    def test_raw_metadata(self):
        """
        TODO: Run a testing suite on all the properties of base raw loader. ensure that they are contained and well defined.
        :return:
        """
        pass

    @property
    def samplerate(self):
        return self.metadata["samplerate"]

    @property
    def cezcontacts(self):
        return self.metadata["ez_hypo_contacts"]

    @property
    def resectedcontacts(self):
        return self.metadata["resected_contacts"]

    @property
    def channel_semiology(self):
        return self.metadata["semiology"]

    @property
    def cezlobe(self):
        return self.metadata["cezlobe"]

    @property
    def record_filename(self):
        return self.metadata["filename"]

    @property
    def bad_channels(self):
        return self.metadata["bad_channels"]

    @property
    def non_eeg_channels(self):
        return self.metadata["non_eeg_channels"]

    @property
    def patient_id(self):
        if "patient_id" in self.metadata.keys():
            return self.metadata["patient_id"]
        else:
            return None

    @property
    def dataset_id(self):
        if "dataset_id" in self.metadata.keys():
            return self.metadata["dataset_id"]
        else:
            return None

    @property
    def meas_data(self):
        return self.metadata["date_of_recording"]

    @property
    def length_of_recording(self):
        return self.metadata["length_of_recording"]

    @property
    def numberchans(self):
        return self.metadata["number_chans"]

    @property
    def clinical_center(self):
        return self.metadata["clinical_center"]

    @property
    def dataset_events(self):
        return self.metadata["events"]

    @property
    def onsetind(self):
        if self.metadata["onset"] is not None:
            return np.multiply(self.metadata["onset"], self.samplerate)
        else:
            return None

    @property
    def offsetind(self):
        if self.metadata["termination"] is not None:
            return np.multiply(self.metadata["termination"], self.samplerate)
        else:
            return None

    @property
    def onsetsec(self):
        return self.metadata["onset"]

    @property
    def offsetsec(self):
        return self.metadata["termination"]

    def loadraw_jsonfile(self, jsonfilepath):
        """
        Function to load in the .json file for a raw dataset.

        :param jsonfilepath: (os.PathLike) is a filepath to the .json file containing dictionary like metadata
        :return: metadata (dict)
        """
        # make sure the json filename passed in is the full path
        metadata = self._loadjsonfile(jsonfilepath)
        return metadata

    def load_rawfif(self, rawfilepath):
        """
        Function to load in the .fif file for a raw dataset.

        :param rawfilepath: (os.PathLike) is a filepath to the .fif file containing the EEG time series data.
        :return: raw (mne.Raw) is a Raw object from the MNE library that contains the raw NxT dataset and metadata.
        """
        # assert that the rawdata file to be loaded is correct and is for this json file
        # if self.record_filename is not None:
        #     assert self.record_filename in rawfilepath

        # extract raw object
        if rawfilepath.endswith(".fif"):
            raw = mne.io.read_raw_fif(rawfilepath, preload=False, verbose=False)
        else:
            raise ValueError(
                "All files read in with eegio need to be preformatted into fif first!"
            )
        return raw

    def _getalljsonfilepaths(self, eegdir):
        """
        Helper function to get all the jsonfile names for
        user to choose one. Returns them in natural sorted order.

        Gets:
         - all file endings with .json extension
         - without '.' as file pretext

        :return: None
        """
        jsonfilepaths = [
            f
            for f in os.listdir(eegdir)
            if f.endswith(".json")
            if not f.startswith(".")
        ]
        self.jsonfilepaths = natsorted(jsonfilepaths)

    def _loadinfodata(self, rawstruct):
        """
        Helper function to load the Info data structure defined in MNE-Python from a Raw data object.
        Sets the low/high pass frequency that is used in the analog band-pass filtering of the data acquisition.

        :param rawstruct: (mne.Raw) the Raw data object from loading a .fif file.
        :return: None
        """
        # set edge freqs that were used in recording
        # Note: the highpass_freq is the max frequency we should be able to see
        # then.
        self.lowpass_freq = rawstruct.info["lowpass"]
        self.highpass_freq = rawstruct.info["highpass"]

        # set line freq
        self.linefreq = rawstruct.info["line_freq"]

        if self.linefreq is None:
            self.linefreq = 60
            print("hard-code setting line freq to 60.", flush=True)
            # raise Exception("You must pass in line freq for this patient, if it is not set in data.")
        print("Set line frequency as: {}".format(self.linefreq), flush=True)

    def create_data_masks(self, bad_channels, non_eeg_channels, chanlabels):
        # get the good channels
        goodchannels = np.setdiff1d(chanlabels, bad_channels)
        goodchannels = np.setdiff1d(goodchannels, non_eeg_channels)
        return goodchannels

    def _create_noneegchannelmask(self, non_eeg_channels, chanlabels):
        """
        Helper method to construct non eeg channels masked array. For every label in 'non_eeg_channels',
        we find the index inside the chanlabels. The eeg channel (i.e. not in non_eeg_channels)
        indices inside chanlabels are returned.

        :param non_eeg_channels: (list) a list of channel label strings.
        :param chanlabels: (list) a list of all the channel label strings.
        :return: noneegchannelmask (list) a list of indices that are NOT noneeg
        """
        # get indices to keep
        noneegchannelmask = chanlabels.copy()
        for idx, ch in enumerate(chanlabels):
            if ch in non_eeg_channels:
                noneegchannelmask[idx] = np.ma.masked
        return noneegchannelmask

    def _create_badchannelmask(self, bad_channels, chanlabels):
        """
        Helper function to construct bad channels mask array. For every label in 'bad_channels',
        we find the index inside the chanlabels. The good eeg channel (i.e. not in bad_channels)
        indices inside chanlabels are returned.

        :param bad_channels: (list) a list of channel label strings.
        :param chanlabels: (list) a list of all the channel label strings.
        :return: badchannelmask (list) a list of indices that are NOT bad
        """
        # get indices to keep
        badchannelmask = chanlabels.copy()
        for idx, ch in enumerate(chanlabels):
            if ch in bad_channels:
                badchannelmask[idx] = np.ma.masked
        return badchannelmask

    def _create_channels_with_coords_mask(self, chanxyzlabels, chanlabels):
        """
        Helper function to create mask on which contacts we have channel coordinates of.

        :param chanxyzlabels: (list) a list of channel label strings that have xyz coordinates identified in the brain.
        :param chanlabels: (list) a list of all the channel label strings.
        :return: xyzdatamask (list) a list of indices that have xyz coordinates identified.
        """
        if len(chanlabels) > 0:
            xyzdatamask = np.array(
                [idx for idx, ch in enumerate(chanxyzlabels) if ch not in chanlabels]
            )
        else:
            xyzdatamask = np.array([])
        return xyzdatamask

    def _create_whitematter_mask(self, chanxyzlabels, contact_regs):
        """
        Helper function to create a mask based on the channel xyz labels we have and the corresponding contact regions
        we have per each of the contacts.

        Brings in contact regions, which are trimmed of white matter.

        :param chanxyzlabels: (list) a list of channel label strings that have xyz coordinates identified in the brain.
        :param contact_regs: (list) a list of channel indices that are found to be in 'white_matter' regions.
        :return: whitemattermask (list) a list of channel indices that are in gray matter (i.e. not in white matter).
        """
        # inserted code here to try to get ez/resect elecs
        # try:
        #     self.cezcontacts = metadata['ez_hypo_contacts']
        #     self.resectedcontacts = metadata['resected_contacts']
        #     self.channel_semiology = metadata['semiology']
        # except:
        #     self.cezcontacts = []
        #     self.resectedcontacts = []
        #     self.channel_semiology = []
        #
        # try:
        #     self.cezlobe = metadata['cezlobe']
        # except:
        #     self.cezlobe = []
        whitemattermask = np.array(
            [idx for idx, ch in enumerate(chanxyzlabels) if idx not in contact_regs]
        )
        return whitemattermask

    def filter_data(self, rawdata, samplerate, linefreq):
        """
        Wrapper function for filtering the data with a bandpass filter [0.5, samplerate / 2],
        and also notch filter that uses the MNE-Python library.

        :param rawdata: (np.ndarray) the NxT raw EEG dataset
        :param samplerate: (float) is the sampling rate in Hz of the EEG data
        :param linefreq: (float) is the line noise frequency (e.g. in USA it is generally 60 Hz and in EU it is 50 Hz)
        :return: rawdata (np.ndarray) the filtered raw dataset
        """
        # the notch filter to apply at line freqs
        linefreq = int(linefreq)  # LINE NOISE OF HZ
        if linefreq != 50 and linefreq != 60:
            raise ValueError(
                "Line frequency for noise should be either 50 or 60 Hz! You passed {}".format(
                    linefreq
                )
            )

        # the bandpass range to pass initial filtering
        freqrange = [0.5]
        freqrange.append(samplerate // 2 - 1)

        # initialize the line freq and its harmonics
        freqs = np.arange(linefreq, samplerate // 2, linefreq)
        # freqs = np.delete(freqs, np.where(freqs > samplerate // 2)[0])
        # freqs = None
        print("filtering at: ", freqs, " and ", freqrange)

        # run bandpass filter and notch filter
        rawdata = mne.filter.notch_filter(
            rawdata, notch_widths=1, Fs=samplerate, freqs=freqs, verbose=False
        )
        rawdata = mne.filter.filter_data(
            rawdata,
            sfreq=samplerate,
            l_freq=freqrange[0],
            h_freq=freqrange[1],
            verbose=False,
        )

        return rawdata
