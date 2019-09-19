from enum import Enum

import numpy as np
from eegio.base.objects.dataset.result_object import Result

deltaband = [0, 4]
thetaband = [4, 8]
alphaband = [8, 13]
betaband = [13, 30]
gammaband = [30, 90]
highgammaband = [90, 500]


class Freqbands(Enum):
    DELTA = deltaband
    THETA = thetaband
    ALPHA = alphaband
    BETA = betaband
    GAMMA = gammaband
    HIGH = highgammaband


class FreqModelResult(Result):
    def __init__(self, mat, freqs, metadata, phasemat=[], freqbands=Freqbands):
        super(FreqModelResult, self).__init__(mat, metadata)
        self.phasemat = phasemat

        self.freqs = freqs
        # initialize the frequency bands
        self.freqbands = freqbands
        # set the frequency index
        self.freqindex = 1

        # create a copy of the matrix
        self.buffmat = self.mat.copy()

    def get_phase(self):
        return self.phasemat

    def get_freqs(self):
        return self.freqs

    def __str__(self):
        return "{} {} Frequency Power Model {}".format(
            self.patient_id, self.dataset_id, self.shape
        )

    def _computefreqindices(self, freqs, freqband):
        """
        Compute the frequency indices for this frequency band [lower, upper].

        freqs = list of frequencies
        freqband = [lowerbound, upperbound] frequencies of the
                frequency band
        """
        lowerband = freqband[0]
        upperband = freqband[1]

        # get indices where the freq bands are put in
        freqbandindices = np.where((freqs >= lowerband) & (freqs < upperband))
        freqbandindices = [freqbandindices[0][0], freqbandindices[0][-1]]
        return freqbandindices

    def compress_freqbands(self, freqbands=Freqbands):
        # ensure power is absolute valued
        power = np.abs(self.mat)

        # create empty binned power
        power_binned = np.zeros(shape=(power.shape[0], len(freqbands), power.shape[2]))
        print(power.shape, power_binned.shape)
        for idx, freqband in enumerate(freqbands):
            # compute the freq indices for each band
            freqbandindices = self._computefreqindices(self.freqs, freqband.value)

            # Create an empty array = C x T (frequency axis is compresssed into 1 band)
            # average between these two indices
            power_binned[:, idx, :] = np.nanmean(
                power[:, freqbandindices[0] : freqbandindices[1] + 1, :], axis=1
            )

        self.mat = power_binned
        self.freqbands = freqbands
        self.freqbandslist = list(freqbands)

    def format_dataset(self, result, freqindex=None, apply_trim=True, is_scalp=False):
        # number of seconds to offset trim timeseries by
        offset_sec = 10
        # threshold level
        threshlevel = 0.7

        # get channels separated data by cez and others
        if is_scalp:
            # compute montage group
            result.compute_montage_groups()

            print(result.get_data().shape)
            # print(result.chanlabels)
            # print(result.cezlobe)
            # print(len(result.cezlobeinds), len(result.oezlobeinds))

            # partition into cir and oir
            clinonset_map = result.get_data()[result.cezlobeinds, ...]
            others_map = result.get_data()[result.oezlobeinds, ...]
        else:
            clinonset_map = result.get_data()[result.cezinds]
            others_map = result.get_data()[result.oezinds]

        if freqindex is not None:
            clinonset_map = clinonset_map[:, freqindex, :]
            others_map = others_map[:, freqindex, :]
            # print(clinonset_map.shape, others_map.shape)

        if apply_trim:
            # trim timeseries in time
            clinonset_map = result.trim_aroundseizure(
                offset_sec=offset_sec, mat=clinonset_map
            )
            others_map = result.trim_aroundseizure(
                offset_sec=offset_sec, mat=others_map
            )

        clinonset_map = np.mean(clinonset_map, axis=0)
        others_map = np.mean(others_map, axis=0)

        return clinonset_map, others_map
