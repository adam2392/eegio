import os

import numpy as np
import pandas as pd

from eztrack.eegio.objects.dataset.elecs import Contacts
from eztrack.eegio.loaders.dataset.neuroimage.read_connectivity import LoadConn
from eztrack.eegio.loaders.dataset.neuroimage.read_surf import LoadSurface
from eztrack.eegio.utils import utils


class BaseNeuralLoaders(object):
    """
    Class wrapper for basic loading functions to load in the data
    from files generated in the preformat stage.

    It helps load in gain matrix for TVB generation, clinical data, contacts file,
    label volume file, ez hypothesis file, connectivity file, surface file.
    """

    def _loadgainmat(self, gainfile):
        """
        Load in the gain matrix file that projects atlas brain regions -> sensor locations.
        :param gainfile:
        :return:
        """
        if not os.path.exists(gainfile):
            return None
        # function to get model in its equilibrium value
        gain_pd = pd.read_csv(gainfile, header=None, delim_whitespace=True)
        gainmat = gain_pd.as_matrix()
        return gainmat

    def load_clinicalfile(self, clinicalfile):
        """
        Loads in the clinical file excel sheet as a dataframe.
        :param clinicalfile:
        :return:
        """
        # get the df of the clinical data
        clinical_df = pd.read_csv(clinicalfile)

        return clinical_df

    def _loadcontacts(self, sensorsfile):
        """
        Function to load in the contacts.
        :param sensorsfile:
        :return:
        """
        contacts = Contacts()
        contacts.load_from_file(sensorsfile)
        return contacts

    def _loadseegxyz(self, sensorsfile):
        """
        Loads in the channel xyz coordinates.

        :param sensorsfile:
        :return:
        """
        if not os.path.exists(sensorsfile):
            return None
        contacts = self._loadconnectivity(sensorsfile)
        chanxyzlabels = contacts.chanlabels
        chanxyz = contacts.xyz

        return chanxyz, chanxyzlabels

    def _mapcontacts_toregs(self, label_volume_file, sensorsfile):
        """
        Function to map contacts in T1 MRI space to regions based on a region_volume_labeling file that assigns
        points in space to a brain region atlas.

        :param label_volume_file:
        :param sensorsfile:
        :return:
        """
        if not os.path.exists(label_volume_file) or not os.path.exists(sensorsfile):
            return None

        contact_regs = np.array(utils.mapcontacts_toregs(sensorsfile,
                                                         label_volume_file))
        return contact_regs

    def _loadezhypothesis(self, ez_hyp_file):
        """
        Function to load in an EZ hypothesis excel file. Loads in based on regions and then parses
        for a 1 -> index of the EZ hypothesis in brain region.

        :param ez_hyp_file:
        :return:
        """
        if not os.path.exists(ez_hyp_file):
            return None

        reginds = pd.read_csv(ez_hyp_file, header=None,
                              delimiter='\n').as_matrix()
        regezinds = np.where(reginds == 1)[0]
        return regezinds

    def _loadconnectivity(self, connfile):
        """
        Function to load in a connectivity file that defines how each brain region is connected (structural conn.).
        This file is atlas dependent.

        :param connfile:
        :return:
        """
        if not os.path.exists(connfile):
            return None

        conn = LoadConn().readconnectivity(connfile)
        return conn

    def _loadsurface(self, surfacefile, regionmapfile):
        """
        Function to load surface information about the brain. Requires a surface file and a region mapping file
        that maps surface elements (rows in file) to a brain region.

        :param surfacefile:
        :param regionmapfile:
        :return:
        """
        if not os.path.exists(surfacefile) or not os.path.exists(regionmapfile):
            return None

        surf = LoadSurface().loadsurfdata(surfacefile,
                                          regionmapfile,
                                          use_subcort=True)
        return surf
