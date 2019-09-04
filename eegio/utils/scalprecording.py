import mne
import numpy as np
from mne.selection import (read_selection, _SELECTIONS, _EEG_SELECTIONS,
                           _divide_to_regions)


class ScalpRecording():
    def __init__(self, contacts, data, sampling_rate):
        '''
        contacts            (list of tuples) is a list of all the contact labels and their
                            corresponding number
        data                (np.ndarray) is a numpy array of the seeg data [chans x time]
        sampling_rate       (float) is the sampling rate in Hz
        '''
        # Sort the contacts first
        contacts, indices = zip(*sorted((c, i)
                                        for i, c in enumerate(contacts)))

        # set the contacts
        self.contacts = contacts
        self.ncontacts = len(self.contacts)
        self.contact_names = [str(a) + str(b) for a, b in self.contacts]

        # can't do assertion here if we trim contacts with a list...
        assert data.ndim == 2
        # assert data.shape[0] == self.ncontacts

        self.data = data[indices, :]

        self.sampling_rate = sampling_rate
        nsamples = self.data.shape[1]
        self.t = np.linspace(0, (nsamples - 1) *
                             (1. / self.sampling_rate), nsamples)

        self.electrodes = {}
        for i, (name, number) in enumerate(self.contacts):
            if name not in self.electrodes:
                self.electrodes[name] = []
            self.electrodes[name].append(i)

        self.set_bipolar()


def create_mne_topostruct_from_numpy(datamat, chanlabels, samplerate,
                                     montage='standard_1020', **kwargs):
    # if montage not in mne.
    # create the info struct
    info = mne.create_info(ch_names=chanlabels.tolist(), ch_types=[
                           'eeg'] * len(chanlabels), sfreq=samplerate)

    # create the raw object
    raw = mne.io.RawArray(data=datamat, info=info)

    # read in a standardized montage and set it
    montage = mne.channels.read_montage(montage)
    raw.set_montage(montage=montage)

    return raw


def get_lobes(rawinfo):
    ch_groups = _divide_to_regions(rawinfo, add_stim=False)
    return ch_groups
