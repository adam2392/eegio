import collections
import copy
import warnings
from abc import ABC, abstractmethod
from typing import List, Dict

import mne
import numpy as np
from deprecated import deprecated
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
        The contacts represented by a Contacts object. See Contacts for more info.

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
            times: List,
            contacts: Contacts,
            patientid: str = None,
            datasetid: str = None,
            model_attributes: Dict = None,
    ):
        self.mat = mat
        self.times = times
        self.contacts = contacts
        self.patientid = patientid
        self.datasetid = datasetid
        self.model_attributes = model_attributes

        self.bufftimes = self.times.copy()
        self.buffmat = self.mat.copy()
        self.buffcontacts = copy.deepcopy(self.contacts)

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
        self.mat = self.buffmat.copy()
        self.times = self.bufftimes.copy()
        self.contacts = copy.deepcopy(self.buffcontacts)

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
        print("Trying to sort naturally contacts in result object")
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
        from mne.selection import _divide_to_regions

        # rawinfo = mne.create_info(
        #     ch_names=list(self.chanlabels),
        #     ch_types="eeg",
        #     sfreq=self.samplerate,
        #     montage=self.montage,
        # )

        # get channel groups - hashmap of channel indices
        ch_groups = _divide_to_regions(rawinfo, add_stim=False)

        # add this to class object but with lower-cased keys
        self.ch_groups = {}
        for k, v in ch_groups.items():
            self.ch_groups[k.lower()] = v

        # get the indices of the cez lobe
        self.cezlobeinds = []
        for lobe in self.cezlobe:
            self.cezlobeinds.extend(self.ch_groups[lobe])
        self.oezlobeinds = [
            ind for ind in range(len(self.chanlabels)) if ind not in self.cezlobeinds
        ]

    @abstractmethod
    def pickle_results(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    @deprecated(version="0.1", reason="Not working function.")
    def set_localreference(self, chanlabels=[], chantypes=[]):
        """
        Only applies for SEEG and Strip channels.

        TODO: Can run computation for grids

        http://www.jneurosci.org/content/31/9/3400
        :param chanlabels:
        :return:
        """
        remaining_labels = np.array([])
        # apply to channel labels, if none are passed in
        if len(chanlabels) == 0:
            chanlabels = self.chanlabels
        else:
            # get the remaining labels in the channels
            remaining_labels = np.array(
                [ch for ch in self.chanlabels if ch not in chanlabels]
            )
        n = len(chanlabels)

        """ ASSUME ALL SEEG OR STRIP FOR NOW """
        if chantypes == []:
            chantypes = ["seeg" for i in range(n)]

        if any(chantype not in ["seeg", "grid", "strip"] for chantype in chantypes):
            raise ValueError(
                "Channel types can only be of seeg, grid, or strip. "
                "Make sure you pass in valid channel types! You passed in {}".format(
                    chantypes
                )
            )

        # # first naturally sort contacts
        # natinds = index_natsorted(self.chanlabels)
        # sortedchanlabels = self.chanlabels.copy()
        # sortedchanlabels = sortedchanlabels[natinds]

        # # get end indices on the electrode that don't have a local reference
        # endinds = []
        # for elec in self.electrodes.keys():
        #     endinds.extend([self.electrodes[elec][0], self.electrodes[elec][-1]])

        # create a dictionary to store all key/values of channels/nbrs
        localreferenceinds = collections.defaultdict(list)

        # get all neighboring channels
        for i, (electrode, inds) in enumerate(self.electrodes.items()):
            for ind in inds:
                # get for this index the channel type
                chantype = chantypes[ind]
                chanlabel = self.chanlabels[ind]

                if chantype == "grid":
                    # get all grid channels
                    gridlayout = [self.chanlabels[g_ind] for g_ind in inds]
                    # run 2d neighbors
                    nbrs_chans, nbrs_inds = self._get2d_neighbors(gridlayout, chanlabel)

                else:
                    # get all channels for this electrode
                    electrodechans = [self.chanlabels[elec_ind] for elec_ind in inds]
                    # run 1d neighbors
                    nbrs_chans, nbrs_inds = self._get1d_neighbors(
                        electrodechans, chanlabel
                    )

                # create dictionary of neighbor channels
                localreferenceinds[chanlabel] = nbrs_chans

        # loop through combinations of channels
        # for inds in zip(np.r_[:n - 2], np.r_[1:n - 1], np.r_[2:n]):
        #     # get the names for all the channels
        #     names = [sortedchanlabels[ind] for ind in inds]
        #
        #     # get the electrode, and the number for each channel
        #     elec0, num0 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[0]).groups()
        #     elec1, num1 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[1]).groups()
        #     elec2, num2 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[2]).groups()
        #
        #     # if electrode name matches, and the number are off by 1, then apply bipolar
        #     if elec0 == elec1 and elec1 == elec2 and abs(int(num0) - int(num1)) == 1 and abs(
        #             int(num1) - int(num2)) == 1:
        #         localreferenceinds[inds[1]] = [inds[0], inds[2]]

        # get indices for all the local referenced channels
        self.localreferencedict = localreferenceinds

        # compute leftover channels
        leftoverinds = [
            ind
            for ind, ch in enumerate(chanlabels)
            if ch not in localreferenceinds.keys()
        ]
        self.leftoverchanlabels = self.chanlabels[leftoverinds]

        if remaining_labels.size == 0:
            return self.localreferencedict
        else:
            return self.localreferencedict, self.leftoverchanlabels
