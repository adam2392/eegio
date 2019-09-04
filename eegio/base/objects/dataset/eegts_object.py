import mne
import numpy as np

from eegio.base.objects.elecs import Contacts
from eegio.base.objects.dataset.basedataobject import BaseDataset
from eegio.utils.scalprecording import create_mne_topostruct_from_numpy
from eegio.utils.utils import compute_samplepoints


class EEGTimeSeries(BaseDataset):
    """
    The class for our time series data that inherits a basic dataset functionality.
    The additional components for this are related to the mne.Info data structure

    For example:
        - modality (scalp, ecog, seeg)
        - sampling rate,
        - voltage values range,
        - reference scheme

    Attributes
    ----------
    mat : (np.ndarray)
        The time series EEG timeseries that is NxT

    metadata : (dict)
        The dictionary metadata object.

    Notes
    -----

    Examples
    --------
    >>> from eegio.base.objects.dataset.eegts_object import EEGTimeSeries
    >>> rawdata = np.random.rand((80,100))
    >>> metadata = dict()
    >>> ts = EEGTimeSeries(rawdata, metadata)
    """

    def __init__(self, mat, times, contacts, samplerate, modality,
                 patientid=None, datasetid=None, model_attributes=None,
                 wm_contacts=[]):
        if self.mat.ndim > 2:
            raise ValueError("Time series can not have > 2 dimensions right now."
                             "We assume [C x T] shape, channels x time. ")

        times = np.arange(mat.shape[1]).astype(int)
        super(EEGTimeSeries, self).__init__(mat, times, contacts, patientid, datasetid, model_attributes)

        # default as monopolar reference
        self.reference = 'monopolar'
        self.samplerate = samplerate
        self.modality = modality
        self._create_info()
        self.wm_contacts = wm_contacts

    def __str__(self):
        return "{} {} EEG mat ({}) " \
               "{} seconds".format(
            self.patient_id, self.dataset_id, self.mat.shape, self.len_secs)

    def __repr__(self):
        return "{} {} EEG mat ({}) " \
               "{} seconds".format(
            self.patient_id, self.dataset_id, self.mat.shape, self.len_secs)

    def _create_info(self):
        # create the info struct
        if self.modality == 'scalp' or self.modality == 'ecog':
            modality = 'eeg'
        elif self.modality == 'seeg':
            modality = 'seeg'
        self.info = mne.create_info(ch_names=self.chanlabels.tolist(),
                                    ch_types=[modality] * self.n_contacts,
                                    sfreq=self.samplerate)

    @property
    def length_of_recording(self):
        return len(self.times)

    @property
    def n_contacts(self):
        return len(self.contacts.chanlabels)

    @property
    def len_secs(self):
        """
        Determines the length of the recording in seconds
        :return:
        """
        return self.length_of_recording / float(self.samplerate)

    @property
    def date_of_recording(self):
        if 'date_of_recording' in self.metadata.keys():
            return self.metadata['date_of_recording']
        return None

    def set_common_avg_ref(self):
        """
        Set a common average referencing scheme.

        :return: None
        """
        self.ref_signal = np.mean(self.mat, axis=0)
        self.mat = self.mat - self.ref_signal
        self.reference = 'common_avg'

    def set_reference_signal(self, ref_signal):
        """
        Set a custom reference signal to reference all signals by.

        :param ref_signal: (np.ndarray) [Tx1] vector that is subtracted from all signals
        in our timeseries.
        :return: None
        """
        self.ref_signal = ref_signal
        self.mat = self.mat - self.ref_signal
        self.reference = 'custom'

    def set_bipolar(self, chanlabels=[]):
        """
        Set bipolar montage for the eeg data time series.

        :param chanlabels:
        :return:
        """
        # extract bipolar reference scheme from contacts data structure
        self._bipolar_inds = self.contacts.set_bipolar(chanlabels=chanlabels)

        newmat = []
        for bipinds in self._bipolar_inds:
            newmat.append(self.mat[bipinds[1], :] -
                          self.mat[bipinds[0], :])

        # set the time series to be bipolar
        self.mat = np.array(newmat)
        self.metadata['chanlabels'] = self.chanlabels
        self.reference = 'bipolar'


    def set_local_reference(self, chanlabels=[], chantypes=[]):
        ''' ASSUME ALL SEEG OR STRIP FOR NOW '''
        if chantypes == []:
            chantypes = ['seeg' for i in range(len(self.chanlabels))]

        # extract bipolar reference scheme from contacts data structure
        self.localreferencedict = self.contacts.set_localreference(
            chanlabels=chanlabels, chantypes=chantypes)

        # apply localreference
        for ind, refinds in self.localreferencedict.items():
            if len(refinds) == 2:
                self.mat[ind, :] = self.mat[ind, :] - 0.5 * \
                    (self.mat[refinds[0], :] + self.mat[refinds[1], :])
            elif len(refinds) == 4:
                self.mat[ind, :] = self.mat[ind, :] - 0.25 * (self.mat[refinds[0], :] +
                                                              self.mat[refinds[1], :] +
                                                              self.mat[refinds[2], :] +
                                                              self.mat[refinds[3], :])
        self.reference = 'local'

    def filter_data(self, linefreq, samplerate, bandpass_freq=(0.5, 300)):
        """
        Filters the time series data according to the line frequency (notch) and
        sampling rate (band pass filter).

        :param linefreq:
        :param samplerate:
        :return:
        """
        # the bandpass range to pass initial filtering
        freqrange = [0.5]
        freqrange.append(samplerate // 2 - 1)

        # get the bandpass frequencies
        l_freq, h_freq = bandpass_freq

        # the notch filter to apply at line freqs
        linefreq = int(linefreq)  # LINE NOISE OF HZ
        assert linefreq == 50 or linefreq == 60

        # initialize the line freq and its harmonics
        freqs = np.arange(linefreq, samplerate // 2, linefreq)
        freqs = np.delete(freqs, np.where(freqs > samplerate // 2)[0])

        # run bandpass filter and notch filter
        self.mat = mne.filter.filter_data(self.mat,
                                          sfreq=samplerate,
                                          l_freq=l_freq,
                                          h_freq=h_freq,
                                          # pad='reflect',
                                          verbose=False
                                          )
        self.mat = mne.filter.notch_filter(self.mat,
                                           Fs=samplerate,
                                           freqs=freqs,
                                           verbose=False
                                           )

    def mask_indices(self, mask_inds):
        self.mat = self.mat[mask_inds, :]
        self.contacts.mask_contact_indices(mask_inds)

    def partition_into_windows(self, winsize, stepsize):
        # compute samplepoints that will perform partitionining
        samplepoints = compute_samplepoints(
            winsize, stepsize, self.length_of_recording)

        # partition the time series data into windows
        formatted_data = []

        # loop through and format data into chunks of windows
        for i in range(len(samplepoints)):
            win = samplepoints[i, :].astype(int)
            data_win = self.mat[:, win[0]:win[1]]

            if data_win.shape[1] == winsize:
                # swap the time and channel axis
                data_win = np.moveaxis(data_win, 0, 1)
                # append result
                formatted_data.append(data_win)

        return formatted_data

