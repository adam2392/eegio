# -*- coding: utf-8 -*-
import re
from enum import Enum
import numpy as np

from eegio.base.objects.dataset.basedataobject import BaseDataset
from eegio.base.objects.elecs import Contacts
from eegio.base.utils.data_structures_utils import compute_timepoints, load_szinds


class Result(BaseDataset):
    """
    The class object for our EEG time series data that is transformed from our raw EEG

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
    >>> from eegio.base.objects.dataset.result_object import Result
    >>> simulated_result = np.random.rand((80,100))
    >>> metadata = dict()
    >>> resultobj = Result(simulated_result, metadata)
    """

    def __init__(
        self,
        mat,
        times,
        contacts,
        patientid=None,
        datasetid=None,
        model_attributes=None,
        metadata=dict(),
    ):
        if mat.ndim > 2:
            raise ValueError(
                "Time series can not have > 2 dimensions right now."
                "We assume [C x T] shape, channels x time. "
            )
        if mat.shape[0] != len(contacts):
            matshape = mat.shape
            ncontacts = len(contacts)
            raise AttributeError(
                f"Matrix data should be shaped: Num Contacts X Time. You "
                f"passed in {matshape} and {ncontacts} contacts."
            )

        super(Result, self).__init__(
            mat, times, contacts, patientid, datasetid, model_attributes
        )

        self.metadata = metadata

        # extract metadata for this time series
        self.extract_metadata()

    def __str__(self):
        return "{} {} EEG result mat ({}) " "{} seconds".format(
            self.patientid, self.datasetid, self.mat.shape, self.len_secs
        )

    def __repr__(self):
        return "{} {} EEG result mat ({}) " "{} seconds".format(
            self.patientid, self.datasetid, self.mat.shape, self.len_secs
        )

    def summary(self):
        pass

    def pickle_results(self):
        pass

    @property
    def n_contacts(self):
        return len(self.contacts.chanlabels)

    @property
    def length_of_result(self):
        return len(self.times)

    @property
    def record_filename(self):
        return self.metadata["filename"]

    def create_fake_example(self):
        pass

    def get_metadata(self):
        """
        Getter method for the dictionary metadata.

        :return: metadata (dict)
        """
        return self.metadata

    def get_model_attributes(self):
        """
        Getter method for returning the model attributes applied to get this resulting data.

        :return:
        """
        return self.model_attributes

    def mask_channels(self):
        """
        Function to apply mask to channel labels based on if they are denoted as:

            - bad
            - non_eeg

        Applies the mask to the object's mat and contacts data structure.

        :return: None
        """
        badchannels = self.metadata["bad_channels"]
        noneegchannels = self.metadata["non_eeg_channels"]

        maskinds = []
        for chlist in [badchannels, noneegchannels]:
            removeinds = self.remove_channels(chlist)
            maskinds.extend(removeinds)

        maskinds = list(set(maskinds))
        nonmaskinds = [
            ind for ind in range(len(self.chanlabels)) if ind not in maskinds
        ]

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

    @property
    def timepoints(self):
        # compute time points
        # return self.samplepoints.astype(int) / self.samplerate
        # return self.metadata['timepoints']
        # compute time points
        return compute_timepoints(
            self.samplepoints.ravel()[-1], self.winsize, self.stepsize, self.samplerate
        )

    @property
    def onsetwin(self):
        # if self.metadata['onsetwin'] == []:
        #     return None

        if self.metadata["onsetwin"] is not None and not np.isnan(
            self.metadata["onsetwin"]
        ):
            return int(self.metadata["onsetwin"])
        # else:
        try:
            print(
                "Onset index and offsetindex", self.onsetind, self.samplepoints[-1, :]
            )
            onsetwin, _ = load_szinds(self.onsetind, None, self.samplepoints)
            return int(onsetwin[0])
        except:
            return None

    @property
    def offsetwin(self):
        # if self.metadata['offsetwin'] == []:
        #     return None

        if self.metadata["offsetwin"] is not None and not np.isnan(
            self.metadata["offsetwin"]
        ):
            return int(self.metadata["offsetwin"])
        # else:
        try:
            print(self.offsetind, self.samplepoints[-1, :])
            _, offsetwin = load_szinds(self.onsetind, self.offsetind, self.samplepoints)
            print("Found offsetwin: ", offsetwin)
            return int(offsetwin[0])
        except:
            return None

    def extract_metadata(self):
        """
        Function to extract metadata from the object's dictionary data structure.
        Extracts the

        :return: None
        """
        # self.contacts_list = self.metadata["chanlabels"]
        try:
            # comment out
            self.modality = self.metadata["modality"]
        except Exception as e:
            self.metadata["modality"] = "ieeg"
            self.modality = self.metadata["modality"]
            print("Loading result object. Error in extracting metadata: ", e)

        # # convert channel labels into a Contacts data struct
        # if self.reference == "bipolar" or self.modality == "scalp":
        #     self.contacts = Contacts(self.contacts_list, require_matching=False)
        # else:
        #     self.contacts = Contacts(self.contacts_list)
