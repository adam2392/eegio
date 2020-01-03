from typing import Dict

import numpy as np

from eegio.base.objects.derivatives.basedataobject import BaseDataset
from eegio.base.objects.electrodes.elecs import Contacts


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
    The class object for our EEG time series data that is transformed from our raw EEG.

    All time series are assumed to be in [C x T] shape and use the Contacts
    data structure to handle all contact level functionality.

    Attributes
    ----------
    mat : (np.ndarray)
        The time series EEG dataset that is NxT

    metadata : (dict)
        The dictionary metadata object.

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
        """Summary string."""
        pass

    def create_fake_example(self):
        """Fake example dataset."""
        pass

    @property
    def samplepoints(self):
        """Samplepoints of each window of data."""
        return np.array(self.times)

    @property
    def record_filename(self):
        """Original filename of the dataset."""
        return self.metadata["filename"]

    @property
    def winsize(self):
        """Window size used in the model."""
        return self.model_attributes["winsize"]

    @property
    def stepsize(self):
        """Step size used in the model."""
        return self.model_attributes["stepsize"]

    @property
    def samplerate(self):
        """Sample rate of the original time series."""
        return self.model_attributes["samplerate"]

    @property
    def timepoints(self):
        """Time points corresponding each column of data."""
        return np.divide(self.times, self.samplerate)

    def compute_onsetwin(self, onsetind: int) -> int:
        """
        Compute onset window corresponding to the onset index.

        Parameters
        ----------
        onsetind : int
            index that event onset occurs (e.g. seizure) in raw data time

        Returns
        -------
        onsetwin : int
            index that event onset occurs in the resulting data transformation

        """
        offsetwin = _findtimewins(onsetind, self.samplepoints)
        return offsetwin

    def compute_offsetwin(self, offsetind: int) -> int:
        """
        Compute offset window from the corresponding index.

        Parameters
        ----------
        offsetind :  int
            index that event offset occurs (e.g. seizure ends) in raw data time

        Returns
        -------
        offsetwin : int
            index that event offset occurs in the resulting data transformation

        """
        offsetwin = _findtimewins(offsetind, self.samplepoints)
        return offsetwin
