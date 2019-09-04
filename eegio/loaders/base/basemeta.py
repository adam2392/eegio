import os

import numpy as np

from eegio.loaders.base.baseio import BaseIO
from eegio.loaders.base.baseloaders import BaseNeuralLoaders

"""
TODO: CHANGE LOADERS TO ALLOW DIFFERENT TYPES OF LOADING
1. FROM JSON FILEPATH DIRECTLY
2. FROM PATIENT PATH
3. FROM FIF PATH?
"""


class BaseMetaLoader(BaseIO):
    """
    # generally optional data depending on how patient was analyzed
    # derived from MRI+CT
    chanxyzlabels = np.array([])
    chanxyz = np.array([])
    # derived from MRI+CT+DWI
    contact_regs = np.array([])
    # derived from connectivity
    conn = None
    weights = np.array([])
    tract_lengths = np.array([])
    region_centres = np.array([])
    region_labels = np.array([])
    # surface object
    surf = None
    # ez hypothesis by clinicians
    regezinds = np.array([])

    # default atlas for loading in parcellation
    atlas = dataconfig.atlas

    ez_hyp_file = ''
    connfile = ''
    surfacefile = ''
    label_volume_file = ''

    """

    def __init__(self, root_dir):
        self.atlas = 'dk'

        # generally optional data depending on how patient was analyzed
        # derived from MRI+CT
        self.chanxyzlabels = np.array([])
        self.chanxyz = np.array([])
        # derived from MRI+CT+DWI
        self.contact_regs = np.array([])
        # derived from connectivity
        self.conn = np.array([])
        self.weights = np.array([])
        self.tract_lengths = np.array([])
        self.region_centres = np.array([])
        self.region_labels = np.array([])
        # surface object
        self.surf = np.array([])
        # ez hypothesis by clinicians
        self.regezinds = np.array([])
        # default atlas for loading in parcellation
        self.ez_hyp_file = ''
        self.connfile = ''
        self.surfacefile = ''
        self.label_volume_file = ''

        self.root_dir = root_dir

        # initialize our loader class to load all sorts of raw data
        self.rawloader = BaseNeuralLoaders()

        # run initialization of files
        self._init_files()

    def load_structural_meta(self):
        # load in connectivity
        self.conn = self.rawloader._loadconnectivity(self.connfile)

        # load in surface
        self.surface = self.rawloader._loadsurface(
            self.surfacefile, self.regionmapfile)

        # load in contacts
        # self.rawloader._loadcontacts()
        if os.path.exists(self.sensorsfile):
            # self.logger.error("Can't from {} because doesn't exist".format(self.sensorsfile))
            self.chanxyz, self.chanxyzlabels = self.rawloader._loadseegxyz(
                self.sensorsfile)
        if os.path.exists(self.gainfile):
            # self.logger.error("Can't from {} because doesn't exist".format(self.gainfile))
            self.gainmat = np.array(self.rawloader._loadgainmat(self.gainfile))
        # map contacts to regions using DWI and T1 Parcellation
        if os.path.exists(self.sensorsfile) and os.path.exists(self.label_volume_file):
            # self.logger.error("Can't from {} because doesn't exist".format(self.label_volume_file))
            self.contact_regs = np.array(self.rawloader._mapcontacts_toregs(
                self.label_volume_file, self.sensorsfile))

        return self.conn, self.surface, self.chanxyz, self.chanxyzlabels, self.gainmat, self.contact_regs

    def map_ez_with_gain(self, thresh=0.1):
        # first normalize gain matrix
        gainmat = self.gainmat / np.max(self.gainmat)
        gainmat[gainmat < thresh] = 0

        # initialize region mapping from regions -> contacts
        reg_hash = dict()

        for i in range(gainmat.shape[0]):
            # get current row on result matrix, and find max to gain matrix
            contactreg = np.where(gainmat[i, :] == np.max(gainmat[i, :]))[0][0]

            # add to the reg_matrix
            if contactreg not in reg_hash.keys():
                reg_hash[contactreg] = 1
            else:
                reg_hash[contactreg] += 1
        return reg_hash

    def map_ez_with_regs(self):
        # initialize region mapping from regions -> contacts
        reg_hash = dict()
        for icontact in range(len(self.contact_regs)):
            contactreg = self.contact_regs[icontact] - 1

            if contactreg not in reg_hash.keys():
                reg_hash[contactreg] = 1
            else:
                reg_hash[contactreg] += 1
        return reg_hash

    def load_ez_region_inds(self):
        # load in ez_hypothesis
        self.regezinds = self.rawloader._loadezhypothesis(self.ez_hyp_file)

        return self.regezinds

    def sync_gray_chans(self, contact_regs):
        # reject white matter contacts
        # find channels that are not part of gray matter
        assert len(self.chanxyzlabels) == len(contact_regs)
        # to make sure that our minimum contact is -1 == white matter
        assert np.min(contact_regs) == -1

        _white_channels_mask = np.array(
            [idx for idx, regid in enumerate(contact_regs) if regid == -1])
        return self.chanxyzlabels[_white_channels_mask]
        # return _white_channels_mask

    def _init_files(self):
        """

        Initialization function to be called
        """
        # set the sub directory files
        self.eegdir = os.path.join(self.root_dir, 'seeg', 'fif')
        self.elecdir = os.path.join(self.root_dir, 'elec')
        self.dwidir = os.path.join(self.root_dir, 'dwi')
        self.mridir = os.path.join(self.root_dir, 'mri')
        self.tvbdir = os.path.join(self.root_dir, 'tvb')

        # assumes elec/tvb/dwi/seeg dirs are set
        self._renamefiles()

        # sensors file with xyz coords
        self.sensorsfile = os.path.join(self.elecdir, 'seeg.txt')

        # label volume file for where each contact is
        self.label_volume_file = os.path.join(
            self.dwidir, "label_in_T1.%s.nii.gz" % self.atlas)

        # connectivity file
        self.connfile = os.path.join(
            self.tvbdir, "connectivity.%s.zip" % self.atlas)

        # surface geometry file
        self.surfacefile = os.path.join(
            self.tvbdir, "surface_cort.%s.zip" % self.atlas)

        # region mapping file for the cortical surface
        self.regionmapfile = os.path.join(
            self.tvbdir, "region_mapping_cort.%s.txt" % self.atlas)

        # computed gain matrix file
        self.gainfile = os.path.join(self.elecdir, 'gain_inv-square.txt')

        # get the ez hypothesis file
        self.ez_hyp_file = os.path.join(self.tvbdir, 'ez_hypothesis.txt')

    def _renamefiles(self):
        sensorsfile = os.path.join(self.elecdir, 'seeg.xyz')
        newsensorsfile = os.path.join(self.elecdir, 'seeg.txt')
        try:
            # copyfile(sensorsfile, newsensorsfile)
            os.rename(sensorsfile, newsensorsfile)
        except BaseException:
            print("Already renamed seeg.xyz")
        gainfile = os.path.join(self.elecdir, 'gain_inv-square.mat')
        newgainfile = os.path.join(self.elecdir, 'gain_inv-square.txt')
        try:
            os.rename(gainfile, newgainfile)
        except BaseException:
            print("Already renamed gain.mat")
            # self.logger.debug("\nAlready renamed gain.mat possibly!\n")
        self.sensorsfile = newsensorsfile

    def map_results_to_gain(self, mat, gainmat, thresh=0.1):
        gainmat = gainmat / np.max(gainmat)
        gainmat[gainmat < thresh] = 0

        reg_mat = np.zeros((gainmat.shape[1], mat.shape[1]))
        reg_hash = dict()

        for i in range(gainmat.shape[0]):
            # get current row on result matrix, and find max to gain matrix
            currcontact = mat[i, :]
            contactreg = np.where(gainmat[i, :] == np.max(gainmat[i, :]))[0][0]

            # add to the reg_matrix
            reg_mat[contactreg, :] += currcontact
            if contactreg not in reg_hash.keys():
                reg_hash[contactreg] = 1
            else:
                reg_hash[contactreg] += 1
        # perform averaging if it is the case
        for iregion in reg_hash.keys():
            reg_mat[iregion, :] /= reg_hash[iregion]

        return reg_mat

    def map_index_to_gain(self, ind, gainmat, thresh=0.1):
        gainmat = gainmat / np.max(gainmat)
        gainmat[gainmat < thresh] = 0

        contactreg = np.where(gainmat[ind, :] == np.max(gainmat[ind, :]))[0][0]
        return contactreg
