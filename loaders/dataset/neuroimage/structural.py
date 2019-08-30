import numpy as np

from eztrack.eegio.loaders.base.basemeta import BaseMetaLoader

# to allow compatability between python2/3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


class StructuralDataLoader(object):

    def __init__(self, root_dir, atlas=None):
        # initialize our loader class to load all sorts of raw data
        self.metaloader = BaseMetaLoader(root_dir=root_dir)

        if atlas is not None:
            self.atlas = atlas

        self.root_dir = root_dir

    def load_data(self):
        self.metaloader.load_structural_meta()
        self.metaloader.load_ez_region_inds()

    def sync_xyz_and_raw(self):
        '''             REJECT BAD CHANS LABELED BY CLINICIAN       '''
        # create mask from bad/noneeg channels
        self.badchannelsmask = self.bad_channels
        self.noneegchannelsmask = self.non_eeg_channels

        # create mask from raw recording data and structural data
        if len(self.chanxyzlabels) > 0:
            self.rawdatamask = np.array(
                [ch for ch in self.chanlabels if ch not in self.chanxyzlabels])
        else:
            self.rawdatamask = np.array([])
        if len(self.chanlabels) > 0:
            self.xyzdatamask = np.array(
                [ch for ch in self.chanxyzlabels if ch not in self.chanlabels])
        else:
            self.xyzdatamask = np.array([])

    def create_data_masks(self, chanxyzlabels, contact_regs):
        # create mask from raw recording data and structural data
        # reject white matter contacts
        # find channels that are not part of gray matter
        assert len(chanxyzlabels) == len(contact_regs)
        # to make sure that our minimum contact is -1 == white matter
        assert np.min(contact_regs) == -1

        _white_channels_mask = np.array(
            [idx for idx, regid in enumerate(contact_regs) if regid == -1])
        return chanxyzlabels[_white_channels_mask]
