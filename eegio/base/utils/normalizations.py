import numpy as np
import numpy.matlib


class Normalize:
    @staticmethod
    def compute_fragilitymetric(minnormpertmat):
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape

        # assert N < T
        fragilitymat = np.zeros((N, T))
        for icol in range(T):
            fragilitymat[:, icol] = (
                np.max(minnormpertmat[:, icol]) - minnormpertmat[:, icol]
            ) / np.max(minnormpertmat[:, icol])
        return fragilitymat

    @staticmethod
    def compute_minmaxfragilitymetric(minnormpertmat):
        # get dimensions of the pert matrix
        N, T = minnormpertmat.shape

        # get the min/max for each column in matrix
        minacrosstime = np.min(minnormpertmat, axis=0)
        maxacrosstime = np.max(minnormpertmat, axis=0)

        # normalized data with minmax scaling
        minmax_fragilitymat = -1 * np.true_divide(
            (minnormpertmat - np.matlib.repmat(maxacrosstime, N, 1)),
            np.matlib.repmat(maxacrosstime - minacrosstime, N, 1),
        )
        return minmax_fragilitymat

    @staticmethod
    def compute_znormalized_fragilitymetric(minnormpertmat):
        # get mean, std
        avg_contacts = np.mean(minnormpertmat, keepdims=True, axis=1)
        std_contacts = np.std(minnormpertmat, keepdims=True, axis=1)

        # normalized data with minmax scaling
        return (minnormpertmat - avg_contacts) / std_contacts
