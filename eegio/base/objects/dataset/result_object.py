from typing import Dict

import numpy as np

from eegio.base.objects.dataset.basedataobject import BaseDataset
from eegio.base.objects.elecs import Contacts


def _findtimewins(time, timepoints):
    if time == 0:
        return 0
    else:
        timeind = np.where((time >= timepoints[:, 0]) & (time <= timepoints[:, 1]))[0][
            0
        ]
        return timeind


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

        super(Result, self).__init__(
            mat,
            times=times,
            contacts=contacts,
            patientid=patientid,
            datasetid=datasetid,
            model_attributes=model_attributes,
            metadata=metadata,
        )

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

    def create_fake_example(self):
        pass

    @property
    def samplepoints(self):
        return np.array(self.times)

    @property
    def record_filename(self):
        return self.metadata["filename"]

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

    def compute_onsetwin(self, onsetind):
        offsetwin = _findtimewins(onsetind, self.samplepoints)
        return offsetwin

    def compute_offsetwin(self, offsetind):
        offsetwin = _findtimewins(offsetind, self.samplepoints)
        return offsetwin
