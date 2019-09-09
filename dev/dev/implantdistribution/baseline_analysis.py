import numpy as np


class BaselineAnalysis:
    @classmethod
    def _getmat(self, data, interval, timeaxis=2):
        if timeaxis == 2:
            return data[..., interval]
        elif timeaxis == 0:
            return data[interval, ...]

    @classmethod
    def compute_preictal_baseline(self, data, onsetind, baseline_precursor=0):
        baselinewin = onsetind - baseline_precursor
        baselinemat = data[..., :baselinewin]

        # back-compute in seconds what a window is
        # winsize_secs = winsize / float(samplerate)
        # baselinewin = onsetwin - (baselinePrecursor / winsize_secs)
        return baselinemat

    @classmethod
    def chooseBand(self, result, normalizedData, band='gamma'):
        deltamax = 3
        thetamax = 7.5
        alphamax = 13
        betamax = 30
        gammamax = 100
        freqs = result.get_metadata()['freqs']
        if (band == 'delta'):
            return normalizedData[:, freqs < deltamax, :]
        elif (band == 'theta'):
            return normalizedData[:, np.all([freqs < thetamax, freqs > deltamax], 0), :]
        elif (band == 'alpha'):
            return normalizedData[:, np.all([freqs < alphamax, freqs > thetamax], 0), :]
        elif (band == 'beta'):
            return normalizedData[:, np.all([freqs < betamax, freqs > alphamax], 0), :]
        else:
            return normalizedData[:, freqs > betamax, :]

    @classmethod
    def chooseWindow(self, freqMatrix, start=0, end=-1):
        return freqMatrix[:, start:end]
