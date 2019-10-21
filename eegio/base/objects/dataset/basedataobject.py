import copy
import warnings
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple

import numpy as np
from natsort import order_by_index

from eegio.base.objects.electrodes.elecs import Contacts
from eegio.base.utils.data_structures_utils import ensure_list


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

    cache_data: (bool)
        whether or not to store a copy of the original data passed in.

    metadata: (Dict)
        accompanying metadata related to this dataset

    montage: (List)
        list of xyz coordinates for every contact.

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
        montage: Union[List, str] = None,
    ):
        if metadata is None:
            metadata = dict()

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
        self.montage = montage

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

    def get_metadata(self) -> Dict:
        """
        Getter method for the dictionary metadata.

        Returns
        -------

        """
        return self.metadata

    def get_montage(self) -> str:
        """
        Get's the EEG dataset montage (i.e. xyz coordinates) based on a specific coordinate system. For scalp EEG
        these can be obtained from the a list of set montages in MNE-Python.

        Returns
        -------

        """
        return self.montage

    def update_metadata(self, **kwargs):
        """
        Function to update metadata dictionary with keyword arguments. This method allows the user to flexibly add
        additional metadata attached to the raw EEG dataset. This is then easily exported when the user gets the metadata
        with get_metadata().

        Parameters
        ----------
        kwargs : dict
            keyword arguments to update the metadata dictionary with

        Returns
        -------

        """
        self.metadata.update(**kwargs)

    def remove_element_from_metadata(self, key):
        self.metadata.pop(key)

    def get_model_attributes(self) -> Dict:
        """
        Getter method for returning the model attributes applied to get this resulting data.

        :return:
        """
        return self.model_attributes

    def reset(self):
        """
        Function to cache restore the matrix data, times, and contacts. Requires that user initially cached the data
        with cache_data=True.

        Returns
        -------

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
    def shape(self) -> Tuple:
        return self.mat.shape

    @property
    def contact_tuple_list(self) -> List:
        return [str(a) + str(b) for a, b in self.contacts.chanlabels]

    @property
    def electrodes(self) -> Dict:
        # use a dictionary to store all electrodes
        return self.contacts.electrodes

    @property
    def chanlabels(self) -> np.ndarray:
        return np.array(self.contacts.chanlabels)

    @property
    def xyz_coords(self) -> List:
        return self.contacts.xyz

    @property
    def ncontacts(self) -> int:
        return len(self.chanlabels)

    def natsort_contacts(self):
        """
        Sort out the time series by its channel labels, so they go in
        a natural ordering.

        A1,A2, ..., B1, B2, ..., Z1, Z2, ..., A'1, A'2, ...

        Returns
        -------

        """
        self.buffchanlabels = self.chanlabels.copy()
        natinds = self.contacts.natsort_contacts()
        self.mat = np.array(order_by_index(self.mat, natinds))
        self.metadata["chanlabels"] = np.array(order_by_index(self.chanlabels, natinds))

    def get_data(self) -> np.ndarray:
        """
        Getter function to get the data matrix stored in Dataset.

        Returns
        -------
        mat : np.ndarray
            The data matrix

        """
        return self.mat

    def get_channel_data(self, name, interval=(None, None)) -> np.ndarray:
        idx = list(self.chanlabels).index(name)
        tid1, tid2 = self._interval_to_index(interval)
        return self.mat[idx, tid1:tid2]

    def remove_channels(self, toremovechans) -> List:
        removeinds = [
            ind for ind, ch in enumerate(self.chanlabels) if ch in toremovechans
        ]
        self.contacts.mask_indices(removeinds)
        self.mat = np.delete(self.mat, removeinds, axis=0)
        return removeinds

    def trim_dataset(self, interval=(None, None)) -> np.ndarray:
        """
        Trims dataset to have (seconds) before/after onset/offset.

        If there is no offset, then just takes it offset after onset.

        Parameters
        ----------
        interval : (tuple)
            A specified interval to trim dataset to E.g. (0, 100) will return first 100 time samples of the data.

        Returns
        -------
        mat: (np.ndarray)
            The trimmed data matrix.

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

    def mask_indices(self, mask_inds: Union[List, np.ndarray, int]):
        """
        Function to keep delete certain rows (i.e. channels) of the matrix time series data
        and the labels corresponding to those masked indices.

        Parameters
        ----------
        mask_inds : np.ndarray
            The indices which we will delete rows from the data matrix and the list of contacts.

        Returns
        -------

        """
        self.mat = np.delete(self.mat, mask_inds, axis=0)
        self.contacts.mask_indices(mask_inds)

    def mask_chs(self, chs: Union[List, np.ndarray, str]):
        """
        Function to keep delete certain rows (i.e. channels) of the matrix time series data
        and the labels corresponding to those masked names.

        Parameters
        ----------
        chs : (list, np.ndarray)
            The set of contact labels to delete from data matrix and list of contacts.

        Returns
        -------

        """
        chs_to_remove = set(self.chanlabels).intersection(ensure_list(chs))
        extra_chs = set(self.chanlabels).difference(ensure_list(chs))

        if extra_chs:
            warnings.warn(
                f"You passed in extra channels to remove. But they were "
                f"not in dataset. {extra_chs}"
            )

        # chs = [x.upper() for x in chs]
        keepinds = self.contacts.mask_chs(ensure_list(chs_to_remove))
        self.mat = self.mat[keepinds, ...]
