import numpy as np

from eztrack.eegio.objects.neuroimaging.brainregion import Region


class ContactMapping():
    def __init__(self, chanlabels_list, chanxyz_list):
        self.coords = chanxyz_list
        self.labels = chanlabels_list

    def map_contact_to_brain(self, brainreg_centers):
        brainregions = []

        # create region class for each brain region_center
        for i, region_label in enumerate(brainreg_centers.keys()):
            x, y, z = brainreg_centers[region_label]
            region = Region(x, y, z, label=region_label)
            brainregions.append(region)

        # map each contact to its closest brain region
        for i in range(len(self.labels)):

    def map_brain_to_lobe(self, lobedict):
        pass


def normalize(data, baselineMat):
    normalized = np.copy(data)
    for i in np.arange(np.size(data, 0)):
        for j in np.arange(np.size(data, 1)):
            normalized[i, j, :] = (normalized[i, j, :] - np.median(baselineMat[i, j, :])) / np.std(
                baselineMat[i, j, :])
    return normalized


def removeFloor(data, floor=0):
    posData = np.copy(data)
    posData[posData < floor] = .0000000001
    return posData


def getLobe(result, windowedMat, lobe):
    return windowedMat[result.ch_groups[lobe], :]


def getLobeValues(result, windowedMat):
    lf = getLobe(result, windowedMat, 'left-frontal')
    lp = getLobe(result, windowedMat, 'left-parietal')
    lo = getLobe(result, windowedMat, 'left-occipital')
    lt = getLobe(result, windowedMat, 'left-temporal')
    rt = getLobe(result, windowedMat, 'right-temporal')
    ro = getLobe(result, windowedMat, 'right-occipital')
    rp = getLobe(result, windowedMat, 'right-parietal')
    rf = getLobe(result, windowedMat, 'right-frontal')
    matList = [lf, lp, lo, lt, rt, ro, rp, rf]
    return matList

# def getFreqDistribution(patientIndex):
#     patientloader = SubjectResultsLoader(subjid=allpats[patientIndex],
#                                          preload=True,
#                                          datatype='freq',
#                                          root_dir=resultsdir)
#     TotalLobeFreq = np.zeros(8)
#     dataCount = 0
#     for result in patientloader.get_results():
#         result.compute_montage_groups()
#         data = np.power(10, result.get_data())  # C x F x T
#         bm = getBaselineMat(result, data, baselinePrecursor)  # C x F x (T0)
#         normalizedData = normalize(data, bm)  # C x F x T
#         bandMat = chooseBand(result, normalizedData, myband)  # C x Fi x T
#         bandAvg = np.median(bandMat, 1)  # C x 1 x T
#         windowedMat = chooseWindow(
#             bandAvg, windowstart, windowend)  # C x 1 x Tw
#         lobeList = getLobeValues(result, windowedMat)  # (c1 to c8) x 1 x Tw
#         for i in np.arange(len(lobeList)):
#             lobeList[i] = removeFloor(lobeList[i])
#             lobeList[i] = np.power(lobeList[i], power)
#
#         for i in np.arange(len(lobeList)):
#             AvgFreq = np.mean(lobeList[i])  # mean across times and channels
#             TotalLobeFreq[i] = TotalLobeFreq[i] + AvgFreq
#         dataCount += 1
#     AvgLobeFreq = TotalLobeFreq / dataCount
#     AvgLobeFreq = AvgLobeFreq / np.sum(AvgLobeFreq)
#     return AvgLobeFreq, patientloader
