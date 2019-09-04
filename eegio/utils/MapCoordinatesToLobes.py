import scipy.io
import numpy as np
import nibabel as nib
import math
import csv

### FUNCTIONS ###


def readMat(matfile):
    mat = scipy.io.loadmat(matfile)
    elecf = mat['elecf']
    elecPos = elecf['elecpos']
    elecPos = elecPos[0][0]
    labels = elecf['label'][0][0]
    return elecPos, labels


def readTxt(txtfile):
    with open(txtfile) as inf:
        reader = csv.reader(inf, delimiter=" ")
        labels = np.genfromtxt(txtfile, dtype='str', usecols=0)
        elecPos = np.loadtxt(txtfile, usecols=(1, 2, 3))
    return elecPos, labels


# Helper function for calculating distance
def calcDist(myelecPos, mycenterPos):
    sum = 0
    for i in range(3):
        sum += (myelecPos[i] - mycenterPos[i]) ** 2
    return math.sqrt(sum)


# Determines the 3 nearest neighbor regions
def calcPatientElectrodeCenters(elecPos, cCoords):
    closestCenters = cC2 = cC3 = []
    for i in np.arange(0, len(elecPos)):
        minDist = min2Dist = min3Dist = 1000000000000
        minIndex = min2Index = min3Index = -1
        for j in np.arange(0, len(cCoords)):
            dist = calcDist(elecPos[i], cCoords[j])
            if (dist < minDist):
                min3Dist = min2Dist
                min3Index = min2Index
                min2Dist = minDist
                min2Index = minIndex
                minDist = dist
                minIndex = j
            elif (dist < min2Dist):
                min3Dist = min2Dist
                min3Index = min2Index
                min2Dist = dist
                min2Index = j
            elif (dist < min3Dist):
                min3Dist = dist
                min3Index = j
        closestCenters.append(minIndex)
        cC2.append(min2Index)
        cC3.append(min3Index)
    return closestCenters, cC2, cC3


# Adds contacts to regions. Also returns array corresponding each electrode to its region.
def addElectrodeContacts(cLabels, closestCenters, cC2, cC3, Regions, Elecs):
    regionArray = []
    regionArray2 = []
    regionArray3 = []
    for i in closestCenters:
        for j in np.arange(len(Regions)):
            if Regions[j].region_name == cLabels[i]:
                Regions[j].addElec(Elecs[i], 1)
                Elecs[i].region1 = Regions[j]
                regionArray.append(Regions[j])
                break
    for i in cC2:
        for j in np.arange(len(Regions)):
            if Regions[j].region_name == cLabels[i]:
                Regions[j].addElec(Elecs[i], 2)
                Elecs[i].region2 = Regions[j]
                regionArray2.append(Regions[j])
                break
    for i in cC3:
        for j in np.arange(len(Regions)):
            if Regions[j].region_name == cLabels[i]:
                Regions[j].addElec(Elecs[i], 3)
                Elecs[i].region3 = Regions[j]
                regionArray3.append(Regions[j])
                break
    return regionArray, regionArray2, regionArray3


# Adds a Region object to the correct lobe
def addRegionToLobe(Regions, Lobes, region2lobe, regionlabels):
    for i in np.arange(len(Regions)):
        for j in np.arange(len(region2lobe)):
            found = False
            if Regions[i].region_name == regionlabels[j]:
                found = True
                try:
                    regIndex = int(region2lobe[j])
                    Lobes[regIndex].addRegion(Regions[i])
                except ValueError:
                    found = False
                break
        if not found:
            Lobes[0].addRegion(Regions[i])


# Determines which lobe each of the electrode contacts are in.
def closestLobes(Lobes, closestRegions):
    closestLobes = []
    for i in np.arange(len(closestRegions)):
        for j in np.arange(len(Lobes)):
            if closestRegions[i] in Lobes[j].regions:
                closestLobes.append(Lobes[j])
    return closestLobes


if __name__ == '__main__':
    # from matplotlib import pyplot as plt

    ### CODE ###
    ### DATA ###
    pat = 'la04'
    # file = 'Data/' + pat + '_native_flirt_elec_xyz.txt'  #Txt file
    file = 'Data/' + pat + '_elec_f.mat'  # Mat file
    # load patient here
    elecPos, labels = readMat(file)  # Mat file
    # elecPos, labels = readTxt(file)            #txt file

    wm = nib.load('../../../conformed_space/wm.mgz')
    bm = nib.load('../../../conformed_space/brainmask.nii')

    centersFileDir = 'Data/connectivity_dk/' + pat + 'centres.txt'
    regionInfoDir = '../fs2lobes_cinginc_convert.txt'
    lobeInfoDir = '../fs2lobes_cinginc_labels.txt'
    with open(centersFileDir) as inf:
        reader = csv.reader(inf, delimiter=" ")
        cLabels = np.genfromtxt(centersFileDir, dtype='str', usecols=0)[:]
        cCoords = np.loadtxt(centersFileDir, usecols=(1, 2, 3))
    with open(regionInfoDir) as inf:
        reader = csv.reader(inf, delimiter=" ")
        regionlabels = np.genfromtxt(regionInfoDir, dtype='str', usecols=1)
        region2lobe = np.loadtxt(regionInfoDir, usecols=0)
    with open(lobeInfoDir) as inf:
        reader = csv.reader(inf, delimiter=" ")
        lobeLabs = np.genfromtxt(lobeInfoDir, dtype='str', usecols=1)
        lobeNums = np.loadtxt(lobeInfoDir, usecols=0)

    Elecs = []
    for i in np.arange(len(labels)):
        Elecs.append(Contact(labels[i]))

    Regions = []
    for i in np.arange(len(cCoords)):
        Regions.append(
            Region(cCoords[i][0], cCoords[i][1], cCoords[i][2], cLabels[i]))

    # Determine which region the electrode contacts are in
    closestCenters, cC2, cC3 = calcPatientElectrodeCenters(elecPos, cCoords)
    closestRegions, cR2, cR3 = addElectrodeContacts(
        cLabels, closestCenters, cC2, cC3, Regions, Elecs)

    # Build a list of all of the lobes
    Lobes = []
    for i in np.arange(len(lobeNums)):
        Lobes.append(Lobe(lobeLabs[i]))

    # Add Regions to their corresponding Lobes
    addRegionToLobe(Regions, Lobes, region2lobe, regionlabels)

    # Determine which Lobe the electrode contacts are in
    closestLobes = closestLobes(Lobes, closestRegions)

    ### PLOTTING ###
    LobeFrequencies = []
    for i in np.arange(len(Lobes)):
        LobeFrequencies.append(Lobes[i].elecs)

    LobeFrequencies = np.array(LobeFrequencies)
    outputLobes = [1, 2, 3, 4, 9, 10, 11, 12]
    outputLabels = ['lf', 'lp', 'lo', 'lt', 'rt', 'ro', 'rp', 'rf']

    # plt.figure()
    # plt.bar(outputLabels, LobeFrequencies[outputLobes])
    # plt.title("Distribution of contacts in lobes, patient la04")
    # plt.xlabel("Lobe #")
    # plt.ylabel("Freq")
    #
    # plt.show(block=True)
    # plt.interactive(False)

    LobeFrequencies = LobeFrequencies[outputLobes]
    LobeDistribution = LobeFrequencies / np.sum(LobeFrequencies)
    np.save('Results/' + pat + 'contactDistribution.npy', LobeDistribution)
