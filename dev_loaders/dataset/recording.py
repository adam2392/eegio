import os
import sys
import warnings

import mne
import numpy as np

from eegio.loaders.base.baseraw import BaseRawLoader
from eegio.base.objects import TimeSeries

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class Recording(BaseRawLoader):
    """
    A Recording class wrapper to load all recording ieeg data.

    Pass in a root_dir that is the root directory with data organized as
    - patient
        - seeg
            - edf
            - fif
    or organized as:
    - root_dir
        - .json + .fif files

    jsonfilepath is a filepath to some dataset in the /fif/ subdirectory.

    Examples
    --------
    >>> import numpy as np
    >>> from eegio.dev_loaders.dataset.timeseries.recording import Recording
    >>> jsonfilepath = ""
    >>> root_dir = ""
    >>> recording = Recording(jsonfilepath=jsonfilepath,
    ...                root_dir=root_dir, preload=True)
    >>> # or
    >>> recording = Recording(root_dir=root_dir)
    >>> jsonfilepaths = recording.jsonfilepaths
    >>> recording.loadpipeline(jsonfilepaths[0])
    """

    def __init__(
        self, root_dir, jsonfilepath, preload, apply_mask, reference, datatype
    ):
        super(Recording, self).__init__(root_dir=root_dir)

        # jsonfilepath for this recording, reference scheme
        self.jsonfilepath = jsonfilepath
        self.reference = reference

        # should we apply masks to the data?
        self.apply_mask = apply_mask

        # TODO: Have these datastructs imported somewhere
        self.chanlabels = np.array([])
        self.chanxyzlabels = np.array([])
        self.contact_regs = np.array([])
        self.chanxyz = np.array([[], []])

        # set loading flags for dataset
        self.is_loaded = False

        # initialize metadata to a dictionary
        self.metadata = dict()

        # set reference for this recording and jsonfilepath
        if datatype == "ieeg":
            datatype = "seeg"
        if jsonfilepath is None and os.path.exists(
            os.path.join(self.root_dir, datatype, "fif")
        ):
            self.eegdir = os.path.join(self.root_dir, datatype, "fif")
        elif os.path.exists(os.path.join(self.root_dir, "ieeg", "fif")):
            self.eegdir = os.path.join(self.root_dir, "ieeg", "fif")
        elif os.path.exists(os.path.join(self.root_dir, "seeg", "fif")):
            self.eegdir = os.path.join(self.root_dir, "seeg", "fif")
        else:
            self.eegdir = self.root_dir

        # get a list of all the jsonfilepaths possible
        self._getalljsonfilepaths(self.eegdir)
        print("reference is ", reference)
        if preload and self.jsonfilepath is not None:
            self.loadpipeline(self.jsonfilepath)

    def __repr__(self):
        return "Recording('%s')" % os.path.basename(self.jsonfilepath)

    def reset(self):
        """
        Resetting method to reset loader to initial variables
        :return:
        """
        self.is_loaded = False

    @property
    def summary(self):
        summary_dict = {
            "jsonfilepath": os.path.basename(self.jsonfilepath),
            "rawfilepath": os.path.basename(self.record_filename),
            "length": self.length_of_recording,
            "samplerate": self.samplerate,
            "numberchans": self.numberchans,
            "reference": self.reference,
        }
        return summary_dict

    def extract_recording_metadata(self):
        """
        Pipeline loading function for grabbing metadata elements out of the dictionary metadata object, loading the
        recording Raw dataset and Info data structure.

        :return: None
        """
        # load in the metadata json file
        self.metadata = self.loadraw_jsonfile(self.jsonfilepath)

        if "filename" in self.metadata.keys():
            record_filename = self.metadata["filename"]
        else:
            record_filename = os.path.basename(self.jsonfilepath).replace(
                ".json", "_raw.fif"
            )

        # get the filepath for the record data
        rawfilepath = os.path.join(self.eegdir, record_filename)
        # load in the rawdata fif file
        self.rawstruct = self.load_rawfif(rawfilepath)
        self._loadinfodata(self.rawstruct)

        # set samplerate and append to self.metadata
        samplerate = self.rawstruct.info["sfreq"]
        self.chanlabels = np.array(self.rawstruct.ch_names)
        self.metadata["samplerate"] = samplerate
        self.metadata["chanlabels"] = self.chanlabels
        self.metadata["number_chans"] = len(self.chanlabels)

        # extract out self.metadata
        self.load_metadata(self.metadata)

    def extract_masks(self):
        # create data masks to make sure all data is trimmed correctly
        bad_channels = self.bad_channels
        non_eeg_channels = self.non_eeg_channels
        chanxyzlabels = self.chanxyzlabels
        contactregs = self.contact_regs
        chanxyz = self.chanxyz

        # create data masks
        goodchannels = self.create_data_masks(
            bad_channels, non_eeg_channels, self.chanlabels
        )
        return goodchannels

    def mask_raw_data(self, rawdata, mask):
        rawmask = []
        rawmask.extend([idx for idx, ch in enumerate(self.chanlabels) if ch in mask])
        self.rawmask_inds = rawmask
        self.chanlabels = self.chanlabels[self.rawmask_inds]
        return rawdata[self.rawmask_inds, :]

    def set_reference(self):
        pass

    def load_all(self):
        if not self.is_loaded:
            # extract the data
            rawdata = self.get_data(self.rawstruct)

            # set flag to loaded
            self.is_loaded = True
            self.rawdata = rawdata

        return rawdata

    def get_data(self, rawstruct, winlist=[], chunkind=None):
        """
        Method to gets the data directly from the eeg object.

        :param rawstruct: (mne.Raw) the Raw data object loaded in from MNE-Python
        :param winlist: (optional; list) of windows that partition the window.
        :param chunkind: (int) the index inside winlist that you want to chunk by.
        :return: data (np.ndarray) the NxT recording dataset
        """
        if winlist != [] and chunkind is None:
            raise ValueError(
                "Trying to chunk up the recording dataset! However, you need"
                "to pass in a list of windows, and also a chunk index. You passed in"
                "{} for winlist and {} for chunkind".format(winlist, chunkind)
            )

        if chunkind is not None and winlist == []:
            win = winlist[chunkind]
            data = rawstruct.get_data()[:, win[0] : win[1] + 1]
        else:
            data = rawstruct.get_data()
        return data

    def create_raw_struct(self, eegobject):
        """
        Helper method to create a mne raw object of the time recording series using
        the corresponding eegobject.

        :param eegobject: (EEGObject) the EEGObject data structure to create a mne Raw dataset from.
        :return: mneraw (mne.Io.Raw) a MNE Raw object
        """
        eegts = eegobject.get_data()
        chanlabels = eegobject.chanlabels
        metadata = eegobject.get_metadata()
        samplerate = metadata["samplerate"]

        # create the info struct
        info = mne.create_info(
            ch_names=chanlabels.tolist(),
            ch_types=["eeg"] * len(chanlabels),
            sfreq=samplerate,
        )

        # create the raw object
        mneraw = mne.io.RawArray(data=eegts, info=info)
        return mneraw

    def setbasemetadata(self):
        """
        If the user wants to clip the data, then you can save a separate metadata
        file that contains all useful metadata about the dataset.

        :return: None
        """
        self.metadata["chanlabels"] = self.chanlabels

        # Set data from the mne file object
        self.metadata["samplerate"] = self.samplerate
        self.metadata["lowpass_freq"] = self.lowpass_freq
        self.metadata["highpass_freq"] = self.highpass_freq
        self.metadata["linefreq"] = self.linefreq

        if "onset" in self.metadata.keys():
            self.metadata["onsetsec"] = self.onsetsec
            self.metadata["offsetsec"] = self.offsetsec
            self.metadata["onsetind"] = self.onsetind
            self.metadata["offsetind"] = self.offsetind
        else:
            self.metadata["onsetsec"] = None
            self.metadata["offsetsec"] = None
            self.metadata["onsetind"] = None
            self.metadata["offsetind"] = None

        self.metadata["reference"] = self.reference

        # dataset metadata
        self.metadata["type"] = self.type
        self.metadata["number_chans"] = self.numberchans

        # dataset metadata
        # self.metadata['ez_hypo_contacts'] =

        return self.metadata

    def loadpipeline(self):
        """
        Pipeline loading function that performs the necessary loading functions on a passed in .json filepath.

        :return: ts_object (EEGTimeSeries) data object.
        """
        jsonfilepath = self.jsonfilepath
        self.eegdir = self.root_dir
        # make sure the directory is correctly set
        if self.root_dir not in jsonfilepath:
            jsonfilepath = os.path.join(self.eegdir, jsonfilepath)
            self.jsonfilepath = jsonfilepath

        # extract recording metadata
        self.extract_recording_metadata()

        # extract masks
        goodchannelmask = self.extract_masks()

        # now we load in the actual data
        rawdata = self.load_all()

        # allow user to determine whether or not to apply masks to the dataset before getting it back
        if self.apply_mask:
            # mask up the raw data and corresponding channel labels
            rawdata = self.mask_raw_data(rawdata, goodchannelmask)

        # filter data
        rawdata = self.filter_data(rawdata, self.samplerate, self.linefreq)

        # set metadata now that all rawdata is processed
        metadata = self.setbasemetadata()

        # create time series object
        ts_object = TimeSeries(rawdata, metadata=metadata)

        return ts_object
