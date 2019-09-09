import os

from base.config.dataconfig import IEEG_REFERENCES
from eegio.loaders.dataset.recording import Recording
from eegio.base.objects import SEEGTs


class iEEGRecording(Recording):
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

    remove_wm_contacts : bool
        Should the class remove all contacts deemed to be white matter?

    reference : str
        The reference scheme to apply to the scalp EEG (monopolar, bipolar, common_avg).

    Notes
    -----

    In future work, we want to include better referencing schemes.

    Examples
    --------
    >>> import numpy as np
    >>> from eegio.dev.timeseries.timeseries.ieegrecording import iEEGRecording
    >>> jsonfilepath = ""
    >>> root_dir = ""
    >>> recording = iEEGRecording(jsonfilepath=jsonfilepath,
    ...                root_dir=root_dir, preload=True,
    ...                apply_mask=True, remove_wm_contacts=True)
    >>> # or
    >>> recording = iEEGRecording(root_dir=root_dir)
    >>> jsonfilepaths = recording.jsonfilepaths
    >>> recording.loadpipeline(jsonfilepaths[0])
    """

    def __init__(self, root_dir,
                 jsonfilepath=None,
                 preload=False,
                 apply_mask=True,
                 remove_wm_contacts=True,
                 reference='monopolar'):
        if not IEEG_REFERENCES.has_value(reference):
            raise ValueError("Reference has to be one of {}".format(
                list(map(str, IEEG_REFERENCES))))

        super(iEEGRecording, self).__init__(root_dir=root_dir,
                                            jsonfilepath=jsonfilepath,
                                            apply_mask=apply_mask,
                                            preload=preload,
                                            reference=reference,
                                            datatype='ieeg')

        self.remove_wm_contacts = remove_wm_contacts

    def loadpipeline(self, jsonfilepath=None):
        """
        Pipeline loading function that performs the necessary loading functions on a passed in .json filepath.

        :return: eeg_object (SEEGTs, or ECOGTs) data object for iEEG data.
        """

        def num_there(s):
            return any(i.isdigit() for i in s)

        if not self.is_loaded:
            if jsonfilepath is None:
                jsonfilepath = self.jsonfilepath

            # make sure the directory is correctly set
            if self.eegdir not in jsonfilepath:
                jsonfilepath = os.path.join(self.eegdir, jsonfilepath)
            self.jsonfilepath = jsonfilepath

            # extract recording metadata
            self.extract_recording_metadata()

            # now we load in the actual data
            rawdata = self.load_all()

            # allow user to determine whether or not to apply masks to the timeseries before getting it back
            if self.apply_mask:
                # extract masks
                mask = self.extract_masks()

                # mask up the raw data and corresponding channel labels
                rawdata = self.mask_raw_data(rawdata, mask)

                # add indices for channels with at least a number
                rawmask = []
                rawmask.extend([idx for idx, ch in enumerate(
                    self.chanlabels) if num_there(ch)])
                self.rawmask_inds = rawmask
                rawdata = rawdata[self.rawmask_inds, :]
                self.chanlabels = self.chanlabels[self.rawmask_inds]

            # filter data
            rawdata = self.filter_data(rawdata, self.samplerate, self.linefreq)

            self.metadata['modality'] = 'ieeg'
            # set metadata now that all rawdata is processed
            self.setbasemetadata()

            # create time series object
            eeg_object = SEEGTs(rawdata, metadata=self.metadata)

            print(self.metadata.keys())
            # if remove wm contacts
            if self.remove_wm_contacts:
                # remove white matter contacts
                print("Still has white matter contacts: ", eeg_object.shape)
                eeg_object.remove_wmcontacts()

                # debug printing
                print("These contacts are identified wm contacts: ",
                      eeg_object.wm_contacts)
                print(len(eeg_object.chanlabels))
                print("Removed white matter contacts: ", eeg_object.shape)
                assert len(eeg_object.chanlabels) == eeg_object.shape[0]

            if self.reference == 'bipolar':
                eeg_object.set_bipolar(eeg_object.chanlabels)
            elif self.reference == 'common_avg':
                eeg_object.set_common_avg_ref()
            return eeg_object
        else:
            raise RuntimeError(
                "You already loaded the data! Run .reset() to reset data")

    def mask_meta_data(self, masks):
        chanxyzlabels = self.chanxyzlabels
        contact_regs = self.contact_regs
        chanxyz = self.chanxyz

        metamask = []
        metamask.extend([idx for idx, ch in enumerate(chanxyzlabels)
                         if all(ch not in mask for mask in masks)])
        chanxyzlabels = chanxyzlabels[metamask]
        contact_regs = contact_regs[metamask]
        chanxyz = chanxyz[metamask, :]
        self.metamask_inds = metamask

        # print("metamask: ", metamask)
        return chanxyzlabels, contact_regs, chanxyz
