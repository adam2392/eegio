import copy
import warnings
from abc import ABC, abstractmethod
from typing import List, Dict, Union

import mne
import numpy as np
from mne.selection import _divide_to_regions
from natsort import order_by_index

from eegio.base.objects.elecs import Contacts


class BaseDataset(ABC):
    """
    The abstract base class for any multi-variate time series EEG dataset. Or resulting time series done on the EEG dataset.
    All time series are assumed to be in [C x T] shape and use the Contacts
    data structure to handle all contact level functionality.

    All datasets have the following characteristics:
    1. multivariate time series data: CxT array
    2. contacts: C electrode contacts, characterized by the Contacts class
    3. patientid: str of the patient identifier
    4. datasetid: str of the dataset identifier
    5. timepoints: list of T time points, can be a list of T tuples if each element was a window of data. See networkanalysis.
    6. model_attributes: dictionary of model attributes applied if a model was applied to the data.

    Attributes
    ----------
    mat : (np.ndarray)
        The dataset that is CxT multivariate time series

    times : (np.ndarray)
        The time samples of the dataset that is Tx1

    contacts: (Contacts)
        The contacts represented by a Contacts object. See Contacts for more metadata.

    patientid: (str)
        patient identifier.

    datasetid: (str)
        dataset identifier

    model_attributes: (dict)
        model attributes of model applied to the data.

    Notes
    -----

    """

    def __init__(
        self,
        mat: np.ndarray,
        times: Union[List, np.ndarray],
        contacts: Contacts,
        patientid: str = None,
        datasetid: str = None,
        model_attributes: Dict = None,
        cache_data: bool = True,
        metadata: Dict = None,
    ):
        if metadata is None:
            metadata = dict()
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

        self.mat = mat
        self.times = times
        self.contacts = contacts
        self.patientid = patientid
        self.datasetid = datasetid
        self.model_attributes = model_attributes
        self.metadata = metadata

        # create cached copies
        self.cache_data = cache_data
        if self.cache_data:
            self.bufftimes = self.times.copy()
            self.buffmat = self.mat.copy()
            self.buffcontacts = copy.deepcopy(self.contacts)
        else:
            self.bufftimes, self.buffmat, self.buffcontacts = None, None, None

    def __len__(self):
        if self.mat.shape[1] != len(self.times):
            warnings.warn(
                f"Times and matrix have different lengths. Their "
                f"respective shapes are: {np.array(self.times).shape}, {self.mat.shape}."
            )
        return self.mat.shape[1]

    @abstractmethod
    def create_fake_example(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    def get_metadata(self):
        """
        Getter method for the dictionary metadata.

        :return: metadata (dict)
        """
        return self.metadata

    def update_metadata(self, **kwargs):
        self.metadata.update(**kwargs)

    def remove_element_from_metadata(self, key):
        self.metadata.pop(key)

    def get_model_attributes(self):
        """
        Getter method for returning the model attributes applied to get this resulting data.

        :return:
        """
        return self.model_attributes

    def get_best_matching_montage(self, chanlabels):
        """
        Get the best matching montage with respect to this montage
        :param chanlabels:
        :return:
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

    def get_montage_channel_indices(self, montage_name, chanlabels):
        """
        Helper function to get the channel indices corresponding to a certain montage
        hardcoded in the MNE-Python framework. For scalp EEG.

        :param montage_name:
        :type montage_name:
        :param chanlabels:
        :type chanlabels:
        :return:
        :rtype:
        """
        # read in montage and strip channel labels not in montage
        montage = mne.channels.read_montage(montage_name)
        montage_chs = [ch.lower() for ch in montage.ch_names]

        # get indices to keep
        to_keep_inds = [idx for idx, ch in enumerate(chanlabels) if ch in montage_chs]

        return to_keep_inds

    def reset(self):
        """
        Function to cache restore the matrix data, times, and contacts.

        :return:
        :rtype:
        """
        if self.cache_data:
            self.mat = self.buffmat.copy()
            self.times = self.bufftimes.copy()
            self.contacts = copy.deepcopy(self.buffcontacts)
        else:
            raise RuntimeError(
                "You can't reset data because you did not cache the data "
                "originally. Reload the data and pass in 'cache_data=True'."
            )

    @property
    def shape(self):
        return self.mat.shape

    @property
    def contact_tuple_list(self):
        return [str(a) + str(b) for a, b in self.contacts.chanlabels]

    @property
    def electrodes(self):
        # use a dictionary to store all electrodes
        return self.contacts.electrodes

    @property
    def chanlabels(self):
        return np.array(self.contacts.chanlabels)

    @property
    def xyz_coords(self):
        return self.contacts.xyz

    @property
    def ncontacts(self):
        return len(self.chanlabels)

    def natsort_contacts(self):
        """
        Sort out the time series by its channel labels, so they go in
        a natural ordering.

        A1,A2, ..., B1, B2, ..., Z1, Z2, ..., A'1, A'2, ...

        :return:
        """
        self.buffchanlabels = self.chanlabels.copy()
        natinds = self.contacts.natsort_contacts()
        self.mat = np.array(order_by_index(self.mat, natinds))
        self.metadata["chanlabels"] = np.array(order_by_index(self.chanlabels, natinds))

    def get_data(self):
        return self.mat

    def get_channel_data(self, name, interval=(None, None)):
        idx = list(self.chanlabels).index(name)
        tid1, tid2 = self._interval_to_index(interval)
        return self.mat[idx, tid1:tid2]

    def remove_channels(self, toremovechans):
        removeinds = [
            ind for ind, ch in enumerate(self.chanlabels) if ch in toremovechans
        ]
        self.contacts.mask_contact_indices(removeinds)
        self.mat = np.delete(self.mat, removeinds, axis=0)
        return removeinds

    def trim_dataset(self, interval=(None, None)):
        """
        Trims dataset to have (seconds) before/after onset/offset.

        If there is no offset, then just takes it offset after onset.

        :param offset:
        :return:
        """
        tid1, tid2 = self._interval_to_index(interval)
        mat = self.mat[..., tid1:tid2]
        return mat

    def _interval_to_index(self, interval):
        tid1, tid2 = 0, -1
        if interval[0] is not None:
            if interval[0] < self.times[0]:
                tid1 = 0
            else:
                tid1 = np.argmax(self.times >= interval[0])
        if interval[1] is not None:
            if interval[1] > self.times[-1]:
                print(self.times[-1], interval)
                return -1
            else:
                tid2 = np.argmax(self.times >= interval[1])
        return (tid1, tid2)

    def time(self, interval=(None, None)):
        tid1, tid2 = self._interval_to_index(interval)
        return self.times[tid1:tid2]

    def compute_montage_groups(self, rawinfo: mne.Info):
        """
        Compute the montage groups for a raw data.

        :return:
        :rtype:
        """
        # get channel groups - hashmap of channel indices
        ch_groups = _divide_to_regions(rawinfo, add_stim=False)

        # add this to class object but with lower-cased keys
        self.ch_groups = {}
        for k, v in ch_groups.items():
            self.ch_groups[k.lower()] = v
        return self.ch_groups
