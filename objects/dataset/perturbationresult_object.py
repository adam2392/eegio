# -*- coding: utf-8 -*-
import numpy as np

from eztrack.eegio.objects.dataset.result_object import Result


class ImpulseResult(Result):
    def __init__(self, impulse_responses, metadata):
        super(ImpulseResult, self).__init__(impulse_responses, metadata)

        if impulse_responses.ndim != 3:
            raise AttributeError(
                "The passed in impulse model needs to have only 3 dimensions!")

        self.json_fields = [
            'onsetwin',
            'offsetwin',
            'resultfilename',
            'winsize',
            'stepsize',
            'radius',
            'perturbtype',
        ]

    def __str__(self):
        return "{} {} Impulse Response Model {}".format(self.patient_id,
                                                        self.dataset_id,
                                                        self.shape)

    @staticmethod
    def compute_fragilitymetric(minnormpertmat):
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape
        # assert N < T
        fragilitymat = np.zeros((N, T))
        for icol in range(T):
            fragilitymat[:, icol] = (np.max(minnormpertmat[:, icol]) - minnormpertmat[:, icol]) / \
                np.max(minnormpertmat[:, icol])
        return fragilitymat

    @staticmethod
    def compute_minmaxfragilitymetric(minnormpertmat):
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape
        # assert N < T
        minmax_fragilitymat = np.zeros((N, T))

        # get the min/max for each column in matrix
        minacrosstime = np.min(minnormpertmat, axis=0)
        maxacrosstime = np.max(minnormpertmat, axis=0)

        # normalized data with minmax scaling
        minmax_fragilitymat = -1 * np.true_divide((minnormpertmat - np.matlib.repmat(maxacrosstime, N, 1)),
                                                  np.matlib.repmat(maxacrosstime - minacrosstime, N, 1))
        return minmax_fragilitymat

    def likelihood_spread(self, impulse_responses):
        """
        Function that takes in a list of impulse responses from a certain node, and rank orders them based on their norm
        responses.

        :param impulse_responses:
        :return:
        """
        # sort from greatest to least
        sorted_inds = np.argsort(impulse_responses, axis=-1, kind='mergesort')

        # apply min-max normalization along the column

        # loop through each node's response to the impulse
        for i, response in enumerate(impulse_responses):
            print(i, response.shape)

    def compute_selfnorm(self):
        """
        Function to compute the norm of the node's own response to its impulse response. This
        is essentially a measure of how unstable a node is to it's own auto response.

        :return:
        """
        pass


class PerturbationResult(Result):
    """
    The class object for our Perturbation Model result.

    We assume a structure of windows over time with a model imposed on the data as a function of that sliding window.
    This also allows overlapping windows and as a consequence overlapping models.


    Attributes
    ----------
    pertmat : (np.ndarray)
        The model's computed minimum norm perturbation over all contacts over time that is NxT

    metadata : (dict)
        The dictionary metadata object.

    Notes
    -----

    Examples
    --------
    >>> from eztrack.eegio.objects.dataset.perturbationresult_object import PerturbationResult
    >>> simulated_result = np.random.rand((80,100))
    >>> metadata = dict()
    >>> resultobj = PerturbationResult(simulated_result, metadata)
    """

    def __init__(self, pertmat, metadata):
        super(PerturbationResult, self).__init__(pertmat, metadata)

        if pertmat.ndim != 2:
            raise AttributeError(
                "The passed in perturbation model needs to have only 2 dimensions!")

        self.json_fields = [
            'onsetwin',
            'offsetwin',
            'resultfilename',
            'winsize',
            'stepsize',
            'radius',
            'perturbtype',
        ]

    @property
    def radius(self):
        return self.metadata['radius']

    @property
    def perturbtype(self):
        return self.metadata['perturbtype']

    def __str__(self):
        return "{} {} Min-Norm Perturbation Model {}".format(self.patient_id,
                                                             self.dataset_id,
                                                             self.shape)


class FragilityModelResult(Result):
    """
    The class wrapper for fragility model, which includes the ltv network model estimated
    and also the perturbation model applied to the ltv model.

    It contains the:

        - ltvn model: NxNxT
        - pert model: NxT
        - fragility metric normalized model: NxT

    Attributes
    ----------
    ltvmodel : (LtvResult)
        The model's computed minimum norm perturbation over all contacts over time that is NxT

    pertmodel : (PerturbationResult)
        The model's computed minimum norm perturbation over all contacts over time that is NxT

    metadata : (dict)
        The dictionary metadata object.

    Notes
    -----

    Examples
    --------
    >>> from eztrack.eegio.objects.dataset.perturbationresult_object import FragilityModelResult
    >>> simulated_result = np.random.rand((80,100))
    >>> metadata = dict()
    >>> resultobj = PerturbationResult(simulated_result, metadata)
    """

    def __init__(self, ltvmodel, pertmodel, metadata):
        self.ltvmodel = ltvmodel
        self.pertmodel = pertmodel
        self.mat = FragilityModelResult.compute_fragilitymetric(pertmodel.mat)
        self.minmax_fragmat = FragilityModelResult.compute_minmaxfragilitymetric(
            pertmodel.mat)

        super(FragilityModelResult, self).__init__(self.mat, metadata)

        self.buffmat = self.mat.copy()

    def __str__(self):
        return "{} {} Fragility Model {}".format(self.patient_id,
                                                 self.dataset_id,
                                                 self.shape)

    def __repr__(self):
        return str(self)

    @property
    def radius(self):
        return self.metadata['radius']

    @property
    def perturbtype(self):
        return self.metadata['perturbtype']

    @staticmethod
    def compute_fragilitymetric(minnormpertmat):
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape
        # assert N < T
        fragilitymat = np.zeros((N, T))
        for icol in range(T):
            fragilitymat[:, icol] = (np.max(minnormpertmat[:, icol]) - minnormpertmat[:, icol]) / \
                np.max(minnormpertmat[:, icol])
        return fragilitymat

    @staticmethod
    def compute_minmaxfragilitymetric(minnormpertmat):
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape
        # assert N < T
        minmax_fragilitymat = np.zeros((N, T))

        # get the min/max for each column in matrix
        minacrosstime = np.min(minnormpertmat, axis=0)
        maxacrosstime = np.max(minnormpertmat, axis=0)

        # normalized data with minmax scaling
        minmax_fragilitymat = -1 * np.true_divide((minnormpertmat - np.matlib.repmat(maxacrosstime, N, 1)),
                                                  np.matlib.repmat(maxacrosstime - minacrosstime, N, 1))
        return minmax_fragilitymat

    @staticmethod
    def compute_znormalized_fragilitymetric(minnormpertmat):
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape

        # get mean, std
        avg_contacts = np.mean(minnormpertmat, keepdims=True, axis=1)
        std_contacts = np.std(minnormpertmat, keepdims=True, axis=1)

        # normalized data with minmax scaling
        return (minnormpertmat - avg_contacts) / std_contacts

    def apply_thresholding_smoothing(self, threshold, mat=None):
        if mat is None:
            mat = self.mat.copy()
            mat[mat < threshold] = 0
            self.mat = mat
        else:
            mat = mat
            mat[mat < threshold] = 0
        return mat
