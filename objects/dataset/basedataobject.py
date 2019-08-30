import warnings

import mne
import numpy as np
import scipy
from deprecated import deprecated
from natsort import order_by_index

from eztrack.eegio.utils.scalprecording import create_mne_topostruct_from_numpy


class BaseDataset(object):
    """
    The base class for any dataset.

    All time series are assumed to be in [C x T] shape and use the Contacts
    data structure to handle all contact level functionality.

    Attributes
    ----------
    mat : (np.ndarray)
        The dataset that is NxT

    times : (np.ndarray)
        The time samples of the dataset that is Tx1

    Notes
    -----

    """

    def __init__(self, mat, times):
        self.mat = mat
        self.times = times

        self.bufftimes = self.times.copy()
        self.buffmat = self.mat.copy()

    def __len__(self):
        return self.mat.shape[1]

    def load_contacts_regs(self, contact_regs, atlas=''):
        self.contacts.load_contacts_regions(contact_regs)
        self.atlas = atlas

    def load_chanxyz(self, chanxyz, coordsystem="T1MRI"):
        """
         Load in the channel's xyz coordinates.

        :param chanxyz:
        :param coordsystem:
        :return:
        """
        if len(chanxyz) != self.ncontacts:
            raise RuntimeError("In loading channels xyz, chanxyz needs to be"
                               "of equal length as the number of contacts in dataset! "
                               "There is a mismatch chanxyz={} vs "
                               "dataset.ncontacts={}".format(
                len(chanxyz), self.ncontacts
            ))
        self.contacts.load_contacts_xyz(chanxyz)
        self.coordsystem = coordsystem

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

    def create_raw_struct(self, usebestmontage=False):
        eegts = self.get_data()
        chanlabels = self.chanlabels
        samplerate = self.samplerate

        # gets the best montage
        best_montage = self.get_best_matching_montage(chanlabels)

        # gets channel indices to keep
        montage_chs = chanlabels
        montage_data = eegts
        if usebestmontage:
            montage_inds = self.get_montage_channel_indices(
                best_montage, chanlabels)
            montage_chs = chanlabels[montage_inds]
            other_inds = [idx for idx, ch in enumerate(
                chanlabels) if ch not in montage_chs]
            montage_data = eegts[montage_inds, :]

            print("Removed these channels: ", chanlabels[other_inds])

        mne_rawstruct = create_mne_topostruct_from_numpy(montage_data, montage_chs,
                                                         samplerate, montage=best_montage)
        return mne_rawstruct

    def get_montage_channel_indices(self, montage_name, chanlabels):
        # read in montage and strip channel labels not in montage
        montage = mne.channels.read_montage(montage_name)
        montage_chs = [ch.lower() for ch in montage.ch_names]

        # get indices to keep
        to_keep_inds = [idx for idx, ch in enumerate(
            chanlabels) if ch in montage_chs]

        return to_keep_inds

    def reset(self):
        self.mat = self.buffmat.copy()
        self.times = self.bufftimes.copy()
        # self.chanlabels = self.buffchanlabels.copy()

    @property
    def shape(self):
        return self.mat.shape

    @property
    def contact_tuple_list(self):
        return [str(a) + str(b) for a, b in self.contacts_list]

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

    def get_cez_hemisphere(self):
        hemisphere = []
        if any('right' in x for x in self.cezlobe):
            hemisphere.append('right')
        if any('left' in x for x in self.cezlobe):
            hemisphere.append('left')

        if len(hemisphere) == 2:
            warnings.warn(
                "Probably can't analyze patients with onsets in both hemisphere in lobes!")
            print(self)

        inds = []
        for hemi in hemisphere:
            for lobe in self.ch_groups.keys():
                if hemi in lobe:
                    inds.extend(self.ch_groups[lobe])

        return hemisphere, inds

    def get_cez_quadrant(self):
        quadrant_map = {
            'right-front': ['right-temporal', 'right-frontal'],
            'right-back': ['right-parietal', 'right-occipital'],
            'left-front': ['left-temporal', 'left-frontal'],
            'left-back': ['left-parietal', 'left-occipital']
        }
        quad1 = ['frontal', 'temporal']
        quad2 = ['parietal', 'occipital']

        print(self.cezlobe)
        quadrant = []
        for lobe in self.cezlobe:
            if 'right' in lobe:
                if any(x in lobe for x in quad1):
                    quadrant.extend(quadrant_map['right-front'])
                if any(x in lobe for x in quad2):
                    quadrant.extend(quadrant_map['right-back'])
            if 'left' in lobe:
                if any(x in lobe for x in quad1):
                    quadrant.extend(quadrant_map['left-front'])
                if any(x in lobe for x in quad2):
                    quadrant.extend(quadrant_map['left-back'])

        if len(quadrant) == 2:
            warnings.warn(
                "Probably can't analyze patients with onsets in all quadrants!")
            print(self)

        print(quadrant)

        inds = []
        for lobe in quadrant:
            inds.extend(self.ch_groups[lobe])
        return quadrant, inds

    def natsort_contacts(self):
        """
        Sort out the time series by its channel labels, so they go in
        a natural ordering.

        A1,A2, ..., B1, B2, ..., Z1, Z2, ..., A'1, A'2, ...

        :return:
        """
        print("Trying to sort naturally contacts in result object")

        self.buffchanlabels = self.chanlabels.copy()
        # pass
        natinds = self.contacts.natsort_contacts()
        self.mat = np.array(order_by_index(self.mat, natinds))
        self.metadata['chanlabels'] = np.array(
            order_by_index(self.chanlabels, natinds))

    def get_data(self):
        return self.mat

    def get_channel_data(self, name, interval=(None, None)):
        idx = self.chanlabels.index(name)
        tid1, tid2 = self.interval_to_index(interval)
        return self.mat[idx, tid1:tid2]

    def remove_channels(self, toremovechans):
        removeinds = [ind for ind, ch in enumerate(
            self.chanlabels) if ch in toremovechans]
        return removeinds

    def split_cez_oez(self, dataset, cez_inds, oez_inds):
        return dataset[cez_inds, :], dataset[oez_inds, :]

    def trim_dataset(self, mat, interval=(None, None)):
        """
        Trims dataset to have (seconds) before/after onset/offset.

        If there is no offset, then just takes it offset after onset.

        :param offset:
        :return:
        """
        tid1, tid2 = self.interval_to_index(interval)
        mat = mat[..., tid1:tid2]
        return mat

    def interval_to_index(self, interval):
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
        tid1, tid2 = self.interval_to_index(interval)
        return self.times[tid1:tid2]

    def compute_montage_groups(self):
        from mne.selection import _divide_to_regions
        rawinfo = mne.create_info(ch_names=list(self.chanlabels),
                                  ch_types='eeg',
                                  sfreq=self.samplerate,
                                  montage=self.montage)

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
        self.oezlobeinds = [ind for ind in range(
            len(self.chanlabels)) if ind not in self.cezlobeinds]

    @deprecated(version="0.1", reason="Applying outside now.")
    def apply_moving_avg_smoothing(self, window_size):
        def movingaverage(interval, window_size):
            window = np.ones(int(window_size)) / float(window_size)
            return np.convolve(interval, window, 'same')

        mat = self.mat.copy()

        # apply moving average filter to smooth out stuff
        smoothed_mat = np.array(
            [movingaverage(x, window_size=window_size) for x in mat])

        # realize that moving average messes up the ends of the window
        smoothed_mat = smoothed_mat[:, window_size // 2:-window_size // 2]

        self.mat = smoothed_mat
        return smoothed_mat

    @deprecated(version="0.1", reason="Applying outside now.")
    def apply_gaussian_kernel_smoothing(self, window_size):
        from eztrack.eegio.utils.post_process import PostProcess

        mat = self.mat.copy()

        # apply moving average filter to smooth out stuff
        smoothed_mat = np.array([PostProcess.smooth_kernel(
            x, window_len=window_size) for x in mat])

        # realize that moving average messes up the ends of the window
        # smoothed_mat = smoothed_mat[:, window_size // 2:-window_size // 2]

        self.mat = smoothed_mat
        return smoothed_mat

    @deprecated(version="0.1", reason="Applying outside now.")
    def timewarp_data(self, mat, desired_len):
        def resample(seq, desired_len):
            """
            Function to resample an individual signal signal.

            :param seq:
            :param desired_len:
            :return:
            """
            # downsample or upsample using Fourier method
            newseq = scipy.signal.resample(seq, desired_len)
            # or apply downsampling / upsampling based
            return np.array(newseq)

        if mat.ndim == 2:
            newmat = np.zeros((mat.shape[0], desired_len))
        elif mat.ndim == 3:
            newmat = np.zeros((mat.shape[0], mat.shape[1], desired_len))
        else:
            raise ValueError(
                "Matrix passed in can't have dimensions other then 2, or 3. Yours has: {}".format(mat.ndim))

        for idx in range(mat.shape[0]):
            seq = mat[idx, ...].squeeze()
            newmat[idx, :] = resample(seq, desired_len)
        return newmat
