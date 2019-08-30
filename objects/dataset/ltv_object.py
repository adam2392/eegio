import numpy as np

from eztrack.eegio.objects.dataset.result_object import Result


class LtvResult(Result):
    """
    The class object for our ECOG time series data.

    A class wrapper for linear-time varying model result. We assume a structure of windows over time with a model
    imposed on the data as a function of that sliding window. This also allows overlapping windows and as a consequence
    overlapping models.


    Attributes
    ----------
    adjmats : (np.ndarray)
        The time series EEG dataset that is NxNxT

    metadata : (dict)
        The dictionary metadata object.

    Notes
    -----

    Examples
    --------
    >>> from eztrack.eegio.objects.dataset.ltv_object import LtvResult
    >>> simulated_result = np.random.rand((80,80,100))
    >>> metadata = dict()
    >>> resultobj = LtvResult(simulated_result, metadata)
    """

    def __init__(self, adjmats, metadata):
        if adjmats.ndim != 3:
            raise AttributeError(
                "The passed in ltv model needs to have 3 dimensions!")

        if adjmats.shape[1] == adjmats.shape[2]:
            if adjmats.shape[0] == adjmats.shape[1]:
                raise RuntimeError(
                    "Need to implement how to roll back axis for LTV result here!")

            # make sure the time axis is rolled to the back
            adjmats = np.moveaxis(adjmats, 0, -1)

        super(LtvResult, self).__init__(adjmats, metadata)

        self.json_fields = [
            'onsetwin',
            'offsetwin',
            'resultfilename',
            'winsize',
            'stepsize',
        ]

    @property
    def stabilizeflag(self):
        return self.metadata['stabilizeflag']

    def __str__(self):
        return "{} {} LTV Model {}".format(self.patient_id,
                                           self.dataset_id,
                                           self.shape)
