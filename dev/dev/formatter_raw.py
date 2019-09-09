import os
import time

import numpy as np

from eegio.format.dev.edfconversion.convertedf import ConvertEDF
from eegio.base.objects.clinical.datasetmeta import DatasetMeta
from eegio.base.utils.format_data_utils import returnindices
from eegio.base.utils.utils import writejsonfile, merge_metadata

import warnings

NON_EEG = ['dc', 'ekg', 'ref', 'emg', "ecg",
           "events", "markers", "mark", "sti014", "stim", "sti"]
PREFORMAT_VARS = [
    'events', 'highpass', 'lowpass',
    'meas_date', 'sfreq', 'line_freq'
]


def get_preset_channels(ch_names, included_indices):
    return np.array(ch_names)[included_indices].tolist()


def run_conversion(edffilepath, pat_id, dataset_id, json_file, datatype, clinical_center, fifdir, save=True):
    # initialize converter
    if datatype == 'seeg':
        edfconverter = ConvertEDFiEEG()
    else:
        edfconverter = ConvertEDFScalp()

    # time the program
    start = time.time()

    print("LOOKING AT: ", edffilepath)
    # load in the dataset and create metadata object
    edfconverter.load_file(filepath=edffilepath,
                           pat_id=pat_id,
                           dataset_id=dataset_id)
    edfconverter.extract_info_and_events(json_events_file=json_file)
    metadata = edfconverter.create_metadata(clinical_center, pat_id=pat_id)

    # declare where to save these files
    newfifname = metadata['filename']
    newfifname = newfifname.replace('_scalp', '')

    newfifpath = os.path.join(fifdir, newfifname)
    newjsonpath = os.path.join(fifdir, newfifpath.replace('_raw.fif', '.json'))

    print(newfifpath)
    print(newjsonpath)
    rawfif, metadata = edfconverter.convert_fif(
        newfifpath, metadata, save=save, replace=True)

    end = time.time()
    print("Done! Time elapsed: {:.2f} secs".format(end - start))

    return rawfif, metadata


class ConvertEDFiEEG(ConvertEDF):
    def convert_metadata(self, clinical_center, save=True, jsonfilepath=None, **kwargs):
        """
        Function for creating the metadata object for a formatted EDF dataset.

        :param clinical_center:
        :param pat_id:
        :return:
        """
        '''         TO BE DEPRECATED: GET CLINICAL CHANNEL DATA       '''
        included_indices, ez_contacts, resect_contacts, outcome, engel_score = self.create_clinical_metadata(self.patientid,
                                                                                                             self.datasetid)
        # create a metadata dictionary
        metadata = dict()

        # get the bad and non eeg channels
        bad_channels, non_eeg_channels = self.create_channel_metadata(
            included_indices=included_indices)

        # creating the final bad channels list
        bad_chans_list = list(set(bad_channels).union(set(non_eeg_channels)))
        if bad_chans_list != self.bad_channels:
            warnings.warn(
                "Hmm the bad channels you specify are not the ones you sent into the .fif dataset?")
        # else:
        self.bad_channels = bad_channels
        self.non_eeg_channels = non_eeg_channels

        # set metadata about the channels
        metadata['bad_channels'] = self.bad_channels
        metadata['non_eeg_channels'] = self.non_eeg_channels
        metadata['channeltypes'] = self.channeltypes

        # create a filepath for the json object
        metadata['filename'] = self.newfilename
        metadata['edffilename'] = self.filename

        print(self.newfilename, self.filename)
        clinical_data_dict = {
            'patient_id': self.patientid,
            'clinical_center': clinical_center,
            'outcome': outcome,
            'engel_score': engel_score,
            'ez_contacts': ez_contacts,
            'resect_contacts': resect_contacts
        }

        dataset_dict = {
            'filename': self.newfilename,
            'date_of_recording': self.rawinfo['meas_date'],
            'length_of_recording': self.raw.n_times,
            'number_chans': len(self.ch_names),
            'onset': self.onset_sec,
            'termination': self.offset_sec,
            'events': self.raw_events
        }

        # create the dataset metadata object
        self.dataset_metadata = DatasetMeta(dataset_id=self.datasetid)

        # set metadata objects
        self.dataset_metadata.set_clinical_data(**clinical_data_dict)
        self.dataset_metadata.set_dataset_data(**dataset_dict)
        self.dataset_metadata.set_bad_channels(self.bad_channels)
        self.dataset_metadata.set_non_eeg_channels(self.non_eeg_channels)

        # create the metadata using the object's metadata
        newmetadata = self.dataset_metadata.create_metadata()

        # merge the metadata
        metadata = merge_metadata(metadata, newmetadata)

        if save:
            # save the formatted metadata json object
            writejsonfile(metadata, jsonfilepath, overwrite=True)

        return metadata

    def create_channel_metadata(self, included_indices=[]):
        """
        Subfunction for creating a channel metadata list of 'bad' and 'non-eeg' channels.

        :param included_indices:
        :return:
        """
        if len(included_indices) > 0:
            included_indices = np.array(included_indices).astype(int)
            # get the bad channels and non_eeg channels
            good_preset_channels = get_preset_channels(
                self.ch_names, included_indices)
            bad_channels = [
                chan.lower() for chan in self.ch_names if chan not in good_preset_channels]
        else:
            bad_channels = []

        # extract non eeg channels based on some rules we set
        non_eeg_channels = [chan.lower() for chan in self.ch_names if any(
            x in chan for x in NON_EEG)]

        # get rid of these channels == 'e'
        non_eeg_channels.extend([ch for ch in self.ch_names if ch == 'e'])

        # do a final filter to determine if there are any bad channels and/or noneeg channels
        bad_channels.extend(self._look_for_bad_channels(self.ch_names))

        return bad_channels, non_eeg_channels

    def create_clinical_metadata(self, pat_id, dataset_id):
        """
        TODO: Use an excel sheet reader to get all this information

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

        pat_id = pat_id if pat_id is not None else self.pat_id

        # get hard included chans -> get the bad chans index
        included_indices, ez_contacts, resect_contacts, engel_score = returnindices(
            pat_id, seiz_id=dataset_id)
        if engel_score == 1:
            outcome = 'success'
        elif engel_score > 1:
            outcome = 'failure'
        elif engel_score == -1:
            outcome = 'na'
        else:
            raise ValueError(
                "Engel score can only be [-1, 1, 2, 3, 4]. You passed {}".format(engel_score))

        if self.datatype == 'scalp':
            included_indices = []

        ez_contacts = [chan.lower() for chan in ez_contacts]
        resect_contacts = [chan.lower() for chan in resect_contacts]

        # subtract by 1 when ummc because they have "event" marker at the beginning of dataset
        # TODO: Figure out how to get that parsed out automatically
        if 'ummc' in pat_id:
            included_indices = np.array(included_indices) + 1

        return included_indices, ez_contacts, resect_contacts, outcome, engel_score


class ConvertEDFScalp(ConvertEDF):
    def convert_metadata(self, pat_id, dataset_id, clinical_center, save=True, jsonfilepath=None, **kwargs):
        """
        Function for creating the metadata object for a formatted EDF dataset.

        :param clinical_center:
        :param pat_id:
        :return:
        """
        '''         TO BE DEPRECATED: GET CLINICAL CHANNEL DATA       '''
        # use dataset id
        _, _, _, outcome, engel_score = self.create_clinical_metadata(
            pat_id, dataset_id)

        # create a metadata dictionary
        metadata = dict()

        print(pat_id, dataset_id)
        '''         GET THE BAD AND NONEEG CHANNELS     '''
        bad_channels, non_eeg_channels = self.create_channel_metadata(
            included_indices=[])
        self.bad_channels = bad_channels
        self.non_eeg_channels = non_eeg_channels

        # set metadata about the channels
        metadata['bad_channels'] = self.bad_channels
        metadata['non_eeg_channels'] = self.non_eeg_channels
        metadata['channeltypes'] = self.channeltypes

        # create a filepath for the json object
        metadata['filename'] = self.newfilename
        metadata['edffilename'] = self.filename

        print(self.newfilename, self.filename)

        clinical_data_dict = {
            'patient_id': pat_id,
            'clinical_center': clinical_center,
            'outcome': outcome,
            'engel_score': engel_score,
        }

        dataset_dict = {
            'filename': self.newfilename,
            'date_of_recording': self.rawinfo['meas_date'],
            'length_of_recording': self.raw.n_times,
            'number_chans': len(self.ch_names),
            'onset': self.onset_sec,
            'termination': self.offset_sec,
            'events': self.raw_events
        }

        # create the dataset metadata object
        self.dataset_metadata = DatasetMeta(dataset_id=dataset_id)

        # set metadata objects
        self.dataset_metadata.set_clinical_data(**clinical_data_dict)
        self.dataset_metadata.set_dataset_data(**dataset_dict)
        self.dataset_metadata.set_bad_channels(self.bad_channels)
        self.dataset_metadata.set_non_eeg_channels(self.non_eeg_channels)

        print("Bad channels list: ", self.bad_channels)
        print("Channel labels list: ", self.ch_names)

        # create the metadata using the object's metadata
        newmetadata = self.dataset_metadata.create_metadata()

        # merge the metadata
        metadata = merge_metadata(metadata, newmetadata)

        if save:
            # save the formatted metadata json object
            writejsonfile(metadata, jsonfilepath, overwrite=True)

        return metadata

    def create_channel_metadata(self, included_indices=[]):
        """
        Subfunction for creating a channel metadata list of 'bad' and 'non-eeg' channels.

        :param included_indices:
        :return:
        """
        if len(included_indices) > 0:
            included_indices = np.array(included_indices).astype(int)
            # get the bad channels and non_eeg channels
            good_preset_channels = get_preset_channels(
                self.ch_names, included_indices)
            bad_channels = [
                chan.lower() for chan in self.ch_names if chan not in good_preset_channels]
        else:
            bad_channels = []

        # extract non eeg channels based on some rules we set
        non_eeg_channels = [chan.lower() for chan in self.ch_names if any(
            x in chan for x in NON_EEG)]

        # get rid of these channels == 'e'
        non_eeg_channels.extend([ch for ch in self.ch_names if ch == 'e'])

        # do a final filter to determine if there are any bad channels and/or noneeg channels
        bad_channels.extend(self._look_for_bad_channels(self.ch_names))

        return bad_channels, non_eeg_channels

    def create_clinical_metadata(self, pat_id, dataset_id):
        """
        TODO: Use an excel sheet reader to get all this information

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

        pat_id = pat_id if pat_id is not None else self.pat_id

        # get hard included chans -> get the bad chans index
        included_indices, ez_contacts, resect_contacts, engel_score = returnindices(
            pat_id, seiz_id=dataset_id)
        if engel_score == 1:
            outcome = 'success'
        elif engel_score > 1:
            outcome = 'failure'
        elif engel_score == -1:
            outcome = 'na'
        ez_contacts = [chan.lower() for chan in ez_contacts]
        resect_contacts = [chan.lower() for chan in resect_contacts]

        # subtract by 1 when ummc because they have "event" marker at the beginning of dataset
        # TODO: Figure out how to get that parsed out automatically
        if 'ummc' in pat_id:
            included_indices = np.array(included_indices) + 1

        return included_indices, ez_contacts, resect_contacts, outcome, engel_score
