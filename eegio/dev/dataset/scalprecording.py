import os

import mne

from base.config.dataconfig import SCALP_REFERENCES
from eegio.loaders.dataset.recording import Recording
from eegio.base.objects import EEGTs
from eegio.base.utils import create_mne_topostruct_from_numpy


class ScalpRecording(Recording):
    """
    A scalp EEG recording class wrapper to load all recording scalp eeg data.

    Pass in a root_dir that is the root directory with data organized as
    - patient
        - seeg
            - edf
            - fif
        - scalp
            - edf
            - fif
    or organized as:
    - root_dir
        - .json + .fif files

    jsonfilepath is a filepath to some timeseries in the /fif/ subdirectory.

    Attributes
    ----------
    root_dir : os.PathLike
        The root directory that datasets will be located.

    jsonfilepath : os.PathLike
        The filepath to the .json file for the scalp EEG recording

    preload : bool
        Should the class run a loading pipeline for the timeseries automatically?

    apply_mask : bool
        Should the class apply all masks to not load in bad, non_eeg, white_matter channels?

    reference : str
        The reference scheme to apply to the scalp EEG (monopolar, common_avg).

    Notes
    -----

    In future work, we want to include better referencing schemes.

    Examples
    --------
    >>> import numpy as np
    >>> from eegio.dev.timeseries.timeseries.scalprecording import ScalpRecording
    >>> jsonfilepath = ""
    >>> root_dir = ""
    >>> recording = ScalpRecording(jsonfilepath=jsonfilepath,
    ...                root_dir=root_dir, preload=True, apply_mask=True)
    >>> # or
    >>> recording = ScalpRecording(root_dir=root_dir)
    >>> jsonfilepaths = recording.jsonfilepaths
    >>> recording.loadpipeline(jsonfilepaths[0])
    """

    def __init__(self, root_dir,
                 jsonfilepath=None,
                 preload=False,
                 apply_mask=True,
                 reference='monopolar'):
        if not SCALP_REFERENCES.has_value(reference):
            raise ValueError("Reference has to be one of {}".format(
                list(map(str, SCALP_REFERENCES))))

        super(ScalpRecording, self).__init__(root_dir=root_dir,
                                             jsonfilepath=jsonfilepath,
                                             preload=preload,
                                             apply_mask=apply_mask,
                                             reference=reference,
                                             datatype='scalp')

    def loadpipeline(self, jsonfilepath=None):
        """
        Pipeline loading function that performs the necessary loading functions on a passed in .json filepath.

        :return: eeg_object (EEGTs) data object of the scalp recording.
        """
        if not self.is_loaded:
            if jsonfilepath is None:
                jsonfilepath = self.jsonfilepath

            # make sure the directory is correctly set
            if self.eegdir not in jsonfilepath:
                jsonfilepath = os.path.join(self.eegdir, jsonfilepath)
            self.jsonfilepath = jsonfilepath

            # extract recording metadata
            self.extract_recording_metadata()

            # extract masks
            mask = self.extract_masks()

            # now we load in the actual data
            rawdata = self.load_all()

            self.montage = None
            # allow user to determine whether or not to apply masks to the timeseries before getting it back
            if self.apply_mask:
                # mask up the raw data and corresponding channel labels
                rawdata = self.mask_raw_data(rawdata, mask)

                # get rid of channels not in montage
                best_montage = self.get_best_matching_montage(self.chanlabels)
                montage_chan_inds = self.get_montage_channel_indices(
                    best_montage, self.chanlabels)
                self.montage = best_montage

                other_inds = [idx for idx in range(
                    len(self.chanlabels)) if idx not in montage_chan_inds]
                print("Removed these channels! ", self.chanlabels[other_inds])

                # get rid of the montage channel indices
                self.chanlabels = self.chanlabels[montage_chan_inds]
                rawdata = rawdata[montage_chan_inds, :]

            # filter data
            rawdata = self.filter_data(rawdata, self.samplerate, self.linefreq)

            self.metadata['montage'] = self.montage
            self.metadata['modality'] = 'scalp'
            self.metadata['cezlobe'] = self.cezlobe
            # set metadata now that all rawdata is processed
            self.setbasemetadata()

            # create time series object
            eeg_object = EEGTs(rawdata, metadata=self.metadata)

            if self.reference == 'common_avg':
                eeg_object.set_common_avg_ref()

            return eeg_object

        else:
            raise RuntimeError(
                "You already loaded the data! Run .reset() to reset data")

    def create_raw_with_montage(self, eegobject, remove=True):
        """
        Function to create a mne.Io.Raw object with a specific montage for scalp EEG timeseries.

        :param eegobject: (EEGTs) an EEG time series data object to convert into a Raw data structure.
        :param remove: (bool) to remove the channels that are not in the predefined MNE montages.
        :return: mne_rawstruct (mne.Io.Raw) the Raw data structure for scalp EEG
        """
        mneraw = self.create_raw_struct(eegobject)

        # gets the best montage
        best_montage = self.get_best_matching_montage(mneraw.ch_names)
        best_montage = 'standard_1020'

        # remove channels not in montage
        if remove:
            montage_inds = self.get_montage_channel_indices(
                best_montage, mneraw.ch_names)
            other_inds = [idx for idx, ch in enumerate(
                mneraw.ch_names) if idx not in montage_inds]
            other_chs = mneraw.ch_names[other_inds]

            # drop these channels
            mneraw.drop_channels(other_chs)
            print("Removed these channels: ", mneraw.ch_names[other_inds])

        # extract data from the channel
        rawdata = mneraw.get_data()
        montage_chs = mneraw.ch_names
        samplerate = mneraw.info['sfreq']

        mne_rawstruct = create_mne_topostruct_from_numpy(
            rawdata, montage_chs, samplerate, montage=best_montage)
        return mne_rawstruct

    def get_montage_channel_indices(self, montage_name, chanlabels):
        """
        Get the channel indices to keep for a specific montage name.

        :param montage_name: (str) a list of predefined montages for scalp EEG recording.
        :param chanlabels: (list) a list of channel label strings.
        :return: to_keep_inds (list) a list of channel indices to keep for this specific montage.
        """
        # read in montage and strip channel labels not in montage
        montage = mne.channels.read_montage(montage_name)
        montage_chs = [ch.lower() for ch in montage.ch_names]

        # get indices to keep
        to_keep_inds = [idx for idx, ch in enumerate(
            chanlabels) if ch in montage_chs]

        return to_keep_inds

    def get_best_matching_montage(self, chanlabels):
        """
        Get the best matching montage with respect to this montage laid out by the channel labels.

        :param chanlabels: (list) a list of channel label strings.
        :return: best_montage (str) is the string of the best montage defined by MNE to use for this list of available
        channel labels.
        """
        montages = mne.channels.get_builtin_montages()
        best_montage = None
        best_montage_score = 0

        for montage_name in montages:
            # read in standardized montage
            montage = mne.channels.read_montage(montage_name)

            # get the channels and score for this montage wrt channels
            montage_score = 0
            montage_chs = [ch.lower() for ch in montage.ch_names]

            # score this montage
            montage_score = len([ch for ch in chanlabels if ch in montage_chs])

            if montage_score > best_montage_score:
                best_montage = montage_name
                best_montage_score = montage_score

        return best_montage
