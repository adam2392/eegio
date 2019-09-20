# -*- coding: utf-8 -*-
import re
from typing import Dict

import numpy as np

from eegio.base.objects.dataset.basedataobject import BaseDataset
from eegio.base.objects.elecs import Contacts
from eegio.base.utils.data_structures_utils import findtimewins


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
            mat: np.ndarray,
            times: np.ndarray,
            contacts: Contacts,
            patientid: str = None,
            datasetid: str = None,
            model_attributes: Dict = None,
            metadata: Dict = None,
    ):
        if metadata is None:
            metadata = {}

        times = list(times)

        super(Result, self).__init__(
            mat,
            times=times,
            contacts=contacts,
            patientid=patientid,
            datasetid=datasetid,
            model_attributes=model_attributes,
        )
        self.metadata = metadata

    def __str__(self):
        return "{} {} EEG result mat ({}) ".format(
            self.patientid, self.datasetid, self.mat.shape
        )

    def __repr__(self):
        return "{} {} EEG result mat ({}) ".format(
            self.patientid, self.datasetid, self.mat.shape
        )

    def summary(self):
        pass

    def pickle_results(self):
        pass

    def create_fake_example(self):
        pass

    @property
    def samplepoints(self):
        return self.times

    @property
    def record_filename(self):
        return self.metadata["filename"]

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

    def mask_channels(self, badchannels):
        """
        Function to apply mask to channel labels based on if they are denoted as:

            - bad
            - non_eeg

        Applies the mask to the object's mat and contacts data structure.

        :return: None
        """
        # badchannels = self.metadata["bad_channels"]
        # noneegchannels = self.metadata["non_eeg_channels"]

        maskinds = []
        for chlist in badchannels:
            removeinds = self.remove_channels(chlist)
            maskinds.extend(removeinds)
        return maskinds

    @property
    def winsize(self):
        return self.model_attributes["winsize"]

    @property
    def stepsize(self):
        return self.model_attributes["stepsize"]

    @property
    def samplerate(self):
        return self.model_attributes["samplerate"]

    @property
    def timepoints(self):
        return np.divide(self.times, self.samplerate)
        # compute time points
        # return compute_timepoints(np.array(self.times).ravel()[-1], self.winsize, self.stepsize, self.samplerate)

    @classmethod
    def compute_onsetwin(self, onsetind):
        offsetwin = findtimewins(onsetind, self.samplepoints)
        return offsetwin

    @classmethod
    def compute_offsetwin(self, offsetind):
        offsetwin = findtimewins(offsetind, self.samplepoints)
        return offsetwin

    @classmethod
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

    @classmethod
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

    @classmethod
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
