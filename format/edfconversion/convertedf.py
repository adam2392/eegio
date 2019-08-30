import os
import re

import mne
import numpy as np
import scipy
import scipy.io

# import pyedflib
from eztrack.base.config.model_constants import DATASET_TYPES
from eztrack.eegio.format.edfconversion.baseconverter import BaseConverter
from eztrack.eegio.objects.dataset.elecs import Contacts
from eztrack.eegio.utils.utils import loadjsonfile


class ConvertEDF(BaseConverter):
    """
    Class wrapper for a data converter from edf files to FiF files.

    Allows the user to:
     - load the edf filepath
     - load in corresponding metadata
     - save and convert the rawdata into .fif format
     - save and convert the metadata into .json format

    Examples
    --------
    >>> from eztrack.eegio.format.edfconversion.convertedf import ConvertEDF
    >>> datatype = 'ieeg'
    >>> filepath = './test.edf'
    >>> converter = ConvertEDF(datatype, filepath)
    >>>
    >>> # or load filepath explicitly
    >>> converter.load_file(filepath)
    >>> newfilepath = './test.fif'
    >>> bad_chans_list = [] # define bad channels
    >>> fif_rawdata = converter.convert_fif(bad_chans_list, newfilepath, save=True, replace=True)
    >>>
    >>> # convert metadata
    >>> clinical_json = {
    ...        'clinical_center': 'nih',
    ...
    ...    }
    >>> metadata = converter.convert_metadata(clinical_json)
    >>> contacts = Contacts(contacts_list, chan_xyz)
    >>> print("These are the contacts: ", contacts)
    """

    def __init__(self, patientid, datasetid, datatype, filepath=None):
        self.raw = None
        self.rawinfo = None
        self.raw_events = dict()
        self.newfilepath = ''
        self.newfilename = ''

        self.patientid = patientid
        self.datasetid = datasetid

        self.bad_channels = []

        # possible datatypes of a eeg dataset
        if datatype in DATASET_TYPES:
            self.datatype = datatype
        else:
            raise ValueError("Datatype can only be one of {}. You passed in {}.".format(
                DATASET_TYPES, datatype))

        if filepath is not None:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.load_file(filepath)

    def load_file(self, filepath):
        """
        Load in a filepath and deconstruct the raw edf file into the mne.io.raw datastruct
        with the info datastructure being decomposed into its parts.

        :param filepath: (os.PathLike) is the filepath to the raw edf file to be converted
        :return: None
        """
        excluded_contacts = [
            "-",
            ""
        ]
        # get the filepath, filename and initialize the reader for the file
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

        # use mne to read the raw edf, events and the info data struct
        self.raw = mne.io.read_raw_edf(
            self.filepath, preload=True, verbose='ERROR', exclude=excluded_contacts)

        # f = pyedflib.EdfReader(fpath)
        # print(f.readAnnotations())
        # pyedflib.close_file(f)

        # store the raw information object
        self.rawinfo = self.raw.info
        # store all the raw edf events
        self.event_times, self.event_ids = mne.events_from_annotations(self.raw)
        self.raw_events['times'] = self.event_times
        self.raw_events['ids'] = self.event_ids

    def extract_info_and_events(self, json_events_file=None, pat_id=None, autofind_markers=True):
        """
        Extracts the events from the edf file. Also, allows user to pass in a json file to get this information.

        -> get the onset/offset indices and seconds from the events.

        :param json_events_file: (optional; os.PathLike) is a filepath to a .json file that has metadata related to this dataset.
        :return:
        """
        # extract data from an information data structure
        self._loadinfodata(self.rawinfo)

        # if pat_id == 'pt11':
        #     def remove_letters(s):
        #         no_digits = []
        #         # Iterate through the string, adding non-numbers to the no_digits list
        #         for i in s:
        #             if i.isdigit():
        #                 no_digits.append(i)
        #
        #         # Now join all elements of the list with '',
        #         # which puts all of the characters together.
        #         result = ''.join(no_digits)
        #         return result
        #
        #     contacts = Contacts(contacts_list=self.ch_names,
        #                         require_matching=False)
        #     # get electrode label for this channel
        #     # get rest of electrode labels
        #     elec_contacts_nums = [int(remove_letters(self.ch_names[ind]))
        #                           for ind in contacts.electrodes["r"]]
        #     print(elec_contacts_nums)
        #     contacts_to_replace = ["r"+str(num) for num in elec_contacts_nums]
        #     for contact in contacts_to_replace:
        #         ch_names = self.ch_names.tolist()
        #         ch_names[ch_names.index(contact)].replace("r", "rg")
        #
        #     self.ch_names = np.array(ch_names)
        #     self.rawinfo['ch_names'] = self.ch_names
        #     print(self.ch_names)
        #     raise Exception("")

        if autofind_markers and "ii" not in self.datasetid:
            # extract the onset/offset seconds and indices
            onset_sec, offset_sec, onset_ind, offset_ind = self.find_onsets_offsets(
                self.sfreq, self.event_times, self.event_ids)

            # load in events using a json event file
            if json_events_file is not None:
                # load in the json file and extract manually input information
                json_events = loadjsonfile(json_events_file)
                onset_sec = json_events['onset']
                offset_sec = json_events['termination']
                try:
                    onset_ind = np.ceil(np.multiply(onset_sec, self.sfreq))
                except TypeError as e:
                    print(e)
                    onset_ind = None
                try:
                    offset_ind = np.ceil(np.multiply(offset_sec, self.sfreq))
                except TypeError as e:
                    print(e)
                    offset_ind = None

            print("Extracted onset and offset as: ", onset_sec, offset_sec)
            # set these information points to the class
            self.onset_sec = onset_sec
            self.offset_sec = offset_sec
            self.onset_ind = onset_ind
            self.offset_ind = offset_ind
        else:
            self.onset_sec = None
            self.offset_sec = None

    def _look_for_bad_channels(self, ch_names):
        """
        Helper function to allow hardcoding of what are "bad channels"

        :param ch_names: (list) a list of str channel labels
        :return: bad_channels (list) of string labels
        """
        # initialize a list to store channel label strings
        bad_channels = []

        # look for channels without letter
        bad_channels.extend(
            [ch for ch in ch_names if not re.search('[a-zA-Z]', ch)])
        # look for channels with '$'
        bad_channels.extend([ch for ch in ch_names if re.search('[$]', ch)])

        if self.datatype == 'ieeg':
            BAD_NAMES = ['fz', 'gz']
            bad_channels.extend([ch for ch in ch_names if ch in BAD_NAMES])

            # look for channels that only have letters - turn off for NIH pt17
            letter_chans = [ch for ch in ch_names if re.search('[a-zA-Z]', ch)]
            bad_channels.extend(
                [ch for ch in letter_chans if not re.search('[0-9]', ch)])

        return bad_channels

    def convert_fif(self, bad_chans_list=[], newfilepath=None, save=True, replace=False):
        """
        Conversion function for the rawdata + metadata into a .fif file format. The accompanying metadata .json
        file will be handled in the convert_metadata() function.

        rawdata.edf -> .fif

        :param bad_chans_list: (optional; list) a list of the bad channels string
        :param newfilepath: (optional; os.PathLike) the file path for the converted fif data
        :param save: (optional; bool) to save the output fif dataset or not
        :param replace: (optional; bool) to overwrite if existing newfilepath is already saved
        :return: formatted_raw (mne.Raw) the raw fif dataset
        """
        if save and newfilepath is None:
            raise AttributeError(
                "Filepath must be passed in if you want to save the data!")

        # create a new information structure
        rawdata = self.raw.get_data(return_times=False)

        # create the info data struct
        preformatted_info = self._addchans_to_raw(self.rawinfo, self.ch_names)
        preformatted_info['bads'] = bad_chans_list

        # perform check on the info data struct
        self.check_info(preformatted_info)

        # save the actual raw array
        formatted_raw = mne.io.RawArray(
            rawdata, preformatted_info, verbose='ERROR')

        # determine channel types and add it into the metadata
        if self.datatype == 'ieeg':
            self.label_channel_types()
        else:
            self.channeltypes = ['eeg'] * len(self.ch_names)

        self.bad_channels = bad_chans_list

        if save:
            fmt = 'single'
            # if rawdata.dtype == 'float64':
            #     fmt = 'double'
            # else:
            #     fmt = 'single'

            self.newfilename = os.path.basename(newfilepath)
            self.newfilepath = newfilepath
            formatted_raw.save(newfilepath,
                               overwrite=replace, fmt=fmt,
                               verbose='ERROR')
        return formatted_raw

    def convert_mat(self, newfilepath, dataset_metadata, replace=False):
        """
        Conversion function for edf file data  into a .mat file format

        rawdata + metadata_dict -> .mat

        :param newfilepath:
        :param dataset_metadata:
        :param replace:
        :return:
        """
        # create a new information structure
        rawdata = self.raw.get_data(return_times=False)
        assert rawdata.shape[0] == dataset_metadata['number_chans']

        # creating the final bad channels list
        bad_chans_list = list(set(self.metadata['bad_channels']).
                              union(set(self.metadata['non_eeg_channels'])))
        preformatted_info = self._addchans_to_raw()
        preformatted_info['bads'] = bad_chans_list

        # perform check on the info data struct
        self.check_info(preformatted_info)

        # convert into dictionary to allow saving into mat file
        info = preformatted_info
        info_dict = {}
        for key, item in info.items():
            if item is None:
                item = []
            info_dict[key] = item

        # save the actual raw array
        if replace is False and os.path.exists(newfilepath):
            raise OSError("Destination filepath for mat data already exists. "
                          "Use overwrite=True to force overwrite.")
        preformatted_info['filename'] = os.path.basename(newfilepath)
        mat_dict = {
            'rawdata': rawdata,
            'info_dict': info_dict,
            'metadata': preformatted_info
        }
        scipy.io.savemat(newfilepath, mat_dict)
        return mat_dict

    def create_clinical_metadata(self, pat_id, dataset_id):
        """
        Subfunction for creating a clinical metadata list of important clinical points:

        - included_indices
        - ez_contacts
        - resect_contacts (ablation or resected)
        - outcome (S/F)
        - engel_score (-1, 1-4)

        :param pat_id:
        :param dataset_id:
        :return:
        """
        raise NotImplementedError("Every conversion from edf method needs to implement how to create the clinical "
                                  "metadata for that type of patient / dataset (e.g. imaging info, clinical hypothesis).")
