from typing import List, Dict, Union

import mne
import numpy as np
from mne.selection import _divide_to_regions


class ScalpMontageHelper:
    @staticmethod
    def compute_montage_groups(rawinfo: mne.Info):
        """
        Compute the montage groups for a raw data and categorizes each electrode into a specified brain lobe region
        based on MNE-Python function _divide_to_regions().

        Parameters
        ----------
        rawinfo : mne.Info
            The Info MNE data structure.

        Returns
        -------
        ch_groups : Dict
            A dictionary, mapping each electrode with a different grouping
        """
        # get channel groups - hashmap of channel indices
        ch_groups = _divide_to_regions(rawinfo, add_stim=False)

        # add this to class object but with lower-cased keys
        eeg_ch_groups = {}
        for k, v in ch_groups.items():
            eeg_ch_groups[k.upper()] = v
        return eeg_ch_groups

    @staticmethod
    def get_best_matching_montage(chanlabels: Union[List, np.ndarray]):
        """
        Get the best matching montage with respect to this montage in MNE-Python.

        Parameters
        ----------
        chanlabels : np.ndarray, List
            list of electrode labels to match against fixed montages.

        Returns
        -------

        """

        # get all the MNE built in montages for scalp EEG data
        montages = mne.channels.get_builtin_montages()

        # initialize a scoring mechanism to determine the best fitting montage
        best_montage = None
        best_montage_score = 0

        for montage_name in montages:
            # read in standardized montage
            montage = mne.channels.make_standard_montage(montage_name)

            # get the channels and score for this montage wrt channels
            montage_score = 0
            montage_chs = [ch for ch in montage.ch_names]

            # score this montage
            montage_score = len([ch for ch in chanlabels if ch in montage_chs])

            if montage_score > best_montage_score:
                best_montage = montage_name
                best_montage_score = montage_score

        return best_montage

    @staticmethod
    def get_montage_channel_indices(
        montage_name: str, chanlabels: Union[List, np.ndarray]
    ):
        """
        Helper function to get the channel indices corresponding to a certain montage
        hardcoded in the MNE-Python framework. For scalp EEG.

        Parameters
        ----------
        montage_name : str
            A hardcoded string setting the montage name to be used. See MNE-Python for more info.

        chanlabels : List, np.ndarray
            A list of labels for the contacts.

        Returns
        -------
        to_keep_inds: List
            a list of indices of the contacts/data matrix to keep based on the montage set.

        """
        # read in montage and strip channel labels not in montage
        montage = mne.channels.make_standard_montage(montage_name)
        montage_chs = [ch.upper() for ch in montage.ch_names]

        # get indices to keep
        to_keep_inds = [idx for idx, ch in enumerate(chanlabels) if ch in montage_chs]

        return to_keep_inds
