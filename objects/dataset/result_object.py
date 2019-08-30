# -*- coding: utf-8 -*-
import re

import numpy as np

from eztrack.eegio.objects.dataset.basedataobject import BaseDataset
from eztrack.eegio.objects.dataset.elecs import Contacts
from eztrack.eegio.utils.utils import compute_timepoints, load_szinds


class Result(BaseDataset):
    """
    The class object for our ECOG time series data.

    All time series are assumed to be in [C x T] shape and use the Contacts
    data structure to handle all contact level functionality.

    Attributes
    ----------
    mat : (np.ndarray)
        The time series EEG dataset that is NxT

    metadata : (dict)
        The dictionary metadata object.

    Notes
    -----

    Examples
    --------
    >>> from eztrack.eegio.objects.dataset.result_object import Result
    >>> simulated_result = np.random.rand((80,100))
    >>> metadata = dict()
    >>> resultobj = Result(simulated_result, metadata)
    """

    def __init__(self, mat, metadata):
        self.metadata = metadata

        # extract metadata for this time series
        self.extract_metadata()

        super(Result, self).__init__(mat=mat, times=self.timepoints[:, 1])

    @property
    def summary(self):
        return self.patient_id, self.dataset_id, self.outcome, self.engel_score, self.clinicaldifficulty

    @property
    def samplerate(self):
        return self.metadata['samplerate']

    @property
    def cezcontacts(self):
        if 'ez_hypo_contacts' in self.metadata.keys():
            return self.metadata['ez_hypo_contacts']

        elif 'clinezelecs' in self.metadata.keys():
            return self.metadata['clinezelecs']
        # if 'ablated_contacts' in self.metadata.keys() and self.metadata['ablated_contacts'] != []:
        #     return self.metadata['ablated_contacts']
        #
        # if 'resected_contacts' in self.metadata.keys() and self.metadata['resected_contacts'] != []:
        #     return self.metadata['resected_contacts']
        else:
            return []

    @property
    def cezinds(self):
        clinonsetinds = [i for i, ch in enumerate(self.chanlabels) if
                         ch in self.cezcontacts]
        return clinonsetinds

    @property
    def oezinds(self):
        oezinds = [i for i, ch in enumerate(self.chanlabels) if
                   ch not in self.cezcontacts]
        return oezinds

    @property
    def surgicalinds(self):
        if self.resectedcontacts != []:
            # print(self.resectedcontacts, len(self.resectedcontacts))
            # print("getting resected contacts")
            inds = [i for i, ch in enumerate(self.chanlabels) if
                    ch in self.resectedcontacts]
        elif self.ablatedcontacts != []:
            # print("getting ablated contacts")
            inds = [i for i, ch in enumerate(self.chanlabels) if
                    ch in self.ablatedcontacts]
        else:
            print(self.resectedcontacts == [])
            print(self.ablatedcontacts == [])
            inds = []
        return inds

    @property
    def nonsurgicalinds(self):
        if self.resectedcontacts != []:
            inds = [i for i, ch in enumerate(self.chanlabels) if
                    ch not in self.resectedcontacts]
        elif self.ablatedcontacts != []:
            inds = [i for i, ch in enumerate(self.chanlabels) if
                    ch not in self.ablatedcontacts]
        else:
            inds = [i for i, ch in enumerate(self.chanlabels)]
        return inds

    @property
    def ablatedcontacts(self):
        if 'ablated_contacts' in self.metadata.keys():
            return self.metadata['ablated_contacts']
        else:
            return []

    @property
    def resectedcontacts(self):
        if 'resected_contacts' in self.metadata.keys():
            return self.metadata['resected_contacts']
        else:
            return []

    @property
    def channel_semiology(self):
        return self.metadata['semiology']

    @property
    def cezlobe(self):
        return self.metadata['cezlobe']

    @property
    def record_filename(self):
        return self.metadata['filename']

    @property
    def bad_channels(self):
        return self.metadata['bad_channels']

    @property
    def non_eeg_channels(self):
        return self.metadata['non_eeg_channels']

    @property
    def patient_id(self):
        if 'patient_id' in self.metadata.keys():
            return self.metadata['patient_id']
        else:
            return None

    @property
    def dataset_id(self):
        if 'dataset_id' in self.metadata.keys():
            return self.metadata['dataset_id']
        else:
            return None

    @property
    def meas_data(self):
        return self.metadata['date_of_recording']

    @property
    def length_of_recording(self):
        return self.metadata['length_of_recording']

    @property
    def numberchans(self):
        return self.metadata['number_chans']

    @property
    def clinical_center(self):
        return self.metadata['clinical_center']

    @property
    def dataset_events(self):
        return self.metadata['events']

    @property
    def onsetind(self):
        if self.metadata['onsetsec'] == []:
            return None
        return self.metadata['onsetsec'] * self.samplerate

    @property
    def offsetind(self):
        if self.metadata['offsetsec'] == []:
            return None
        return self.metadata['offsetsec'] * self.samplerate

    @property
    def onsetsec(self):
        return self.metadata['onsetsec']

    @property
    def offsetsec(self):
        return self.metadata['offsetsec']

    @property
    def resultfilename(self):
        return self.metadata['resultfilename']

    @property
    def chanlabels(self):
        return np.array(self.metadata['chanlabels'])
        # return self.contacts.chanlabels

    @property
    def ncontacts(self):
        return len(self.chanlabels)

    @property
    def numwins(self):
        return len(self.samplepoints)

    @property
    def samplepoints(self):
        return np.array(self.metadata['samplepoints'])

    @property
    def timepoints(self):
        # compute time points
        # return self.samplepoints.astype(int) / self.samplerate
        # return self.metadata['timepoints']
        # compute time points
        return compute_timepoints(self.samplepoints.ravel()[-1],
                                  self.winsize,
                                  self.stepsize,
                                  self.samplerate)

    @property
    def montage(self):
        if self.metadata['modality'] == 'scalp':
            return self.metadata['montage']
        else:
            return None

    @property
    def clinicaldifficulty(self):
        if 'clinical_difficulty' in self.metadata.keys():
            return int(self.metadata['clinical_difficulty'])
        else:
            return None

    @property
    def clinicalmatching(self):
        return int(self.metadata['clinical_match'])

    @property
    def outcome(self):
        if 'outcome' in self.metadata.keys():
            return self.metadata['outcome']
        else:
            return None

    @property
    def engel_score(self):
        if 'engel_score' in self.metadata.keys():
            return int(self.metadata['engel_score'])
        else:
            return None

    @property
    def winsize(self):
        return self.metadata['winsize']

    @property
    def stepsize(self):
        return self.metadata['stepsize']

    @property
    def reference(self):
        return self.metadata['reference']

    @property
    def onsetwin(self):
        # if self.metadata['onsetwin'] == []:
        #     return None

        if self.metadata['onsetwin'] is not None and not np.isnan(self.metadata['onsetwin']):
            return int(self.metadata['onsetwin'])
        # else:
        try:
            print("Onset index and offsetindex",
                  self.onsetind, self.samplepoints[-1, :])
            onsetwin, _ = load_szinds(
                self.onsetind, None, self.samplepoints)
            return int(onsetwin[0])
        except:
            return None

    @property
    def offsetwin(self):
        # if self.metadata['offsetwin'] == []:
        #     return None

        if self.metadata['offsetwin'] is not None and not np.isnan(self.metadata['offsetwin']):
            return int(self.metadata['offsetwin'])
        # else:
        try:
            print(self.offsetind, self.samplepoints[-1, :])
            _, offsetwin = load_szinds(
                self.onsetind, self.offsetind, self.samplepoints)
            print("Found offsetwin: ", offsetwin)
            return int(offsetwin[0])
        except:
            return None

    def get_metadata(self):
        """
        Getter method for the dictionary metadata.

        :return: metadata (dict)
        """
        return self.metadata

    def extract_metadata(self):
        """
        Function to extract metadata from the object's dictionary data structure.
        Extracts the

        :return: None
        """
        self.contacts_list = self.metadata['chanlabels']
        try:
            # comment out
            self.modality = self.metadata['modality']
        except Exception as e:
            self.metadata['modality'] = 'ieeg'
            self.modality = self.metadata['modality']
            print("Loading result object. Error in extracting metadata: ", e)

        # convert channel labels into a Contacts data struct
        if self.reference == 'bipolar' or self.modality == 'scalp':
            self.contacts = Contacts(self.contacts_list, require_matching=False)
        else:
            self.contacts = Contacts(self.contacts_list)

    def mask_channels(self):
        """
        Function to apply mask to channel labels based on if they are denoted as:

            - bad
            - non_eeg

        Applies the mask to the object's mat and contacts data structure.

        :return: None
        """
        badchannels = self.metadata['bad_channels']
        noneegchannels = self.metadata['non_eeg_channels']

        maskinds = []
        for chlist in [badchannels, noneegchannels]:
            removeinds = self.remove_channels(chlist)
            maskinds.extend(removeinds)

        maskinds = list(set(maskinds))
        nonmaskinds = [ind for ind in range(
            len(self.chanlabels)) if ind not in maskinds]

        # apply to relevant data
        self.mat = self.mat[nonmaskinds, ...]
        self.contacts.mask_contact_indices(nonmaskinds)

    def expand_bipolar_chans(self, ch_list):
        if ch_list == []:
            return None

        ch_list = [a.replace("’", "'") for a in ch_list]
        new_list = []
        for string in ch_list:
            if not string.strip():
                continue

            # A1-10
            match = re.match("^([a-z]+)([0-9]+)-([0-9]+)$", string)
            if match:
                name, fst_idx, last_idx = match.groups()

                new_list.extend([name + str(fst_idx), name + str(last_idx)])

        return new_list

    def expand_ablated_chans(self, ch_list):
        if ch_list == []:
            return None

        ch_list = [a.replace("’", "'") for a in ch_list]
        new_list = []
        for string in ch_list:
            if not string.strip():
                continue

            # A1-10
            match = re.match("^([a-z]+)([0-9]+)-([0-9]+)$", string)
            if match:
                name, fst_idx, last_idx = match.groups()

                new_list.extend([name + str(fst_idx), name + str(last_idx)])

        return new_list

    def make_onset_labels_bipolar(self, clinonsetlabels):
        added_ch_names = []
        for ch in clinonsetlabels:
            # A1-10
            match = re.match("^([a-z]+)([0-9]+)$", ch)
            if match:
                name, fst_idx = match.groups()
            added_ch_names.append(name + str(int(fst_idx) + 1))

        clinonsetlabels.extend(added_ch_names)
        clinonsetlabels = list(set(clinonsetlabels))
        return clinonsetlabels
