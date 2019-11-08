import os
from eegio.base.config import EEGIO_DIR

import mne
from mne import io

from mne_bids import write_raw_bids, make_bids_basename, make_bids_folders
from mne_bids.utils import print_dir_tree

class BaseBids(object):
    """
    An instantiator class of the bids format to handle the logistics of creating
    a bids format file structure

    Attributes
    ----------
    patientid: str
        patient identifier used to build the file structure
    datadir: os.PathLike
        The base directory for storing data
    Notes
    -----

    Examples
    --------
    """
    def __init__(
        self,
        patientid: str,
        datadir: os.PathLike,
        modality: str,
    ):
        self.patientid=patientid
        self.datadir=datadir
        self.modality=modality

        if (not os.path.exists(os.path.join(self.datadir, self.patientid)):
            self.make_bids_folder_struct()
            bids_basename = make_bids_basename(subject=patientid, session='one')
            self.bids_basename = bids_basename
        
        # store parameters
        self._store_params()

    @property
    def parameters(self):
        return self.parameter_dict

    def _store_params(self):
        self.parameter_dict.update(
            patientid=self.patientid,
            datadir=self.datadir,
            modality=self.modality,
        )

    def summary(self):
        """
        Print a summary of the patient in the form of a directory tree showing
        all available data
        """
        basepath = os.path.join(self.datadir, self.patientid)
        print_dir_tree(basepath)

    def make_bids_folder_struct(self):
        """
        Creates the empty folder structure for a new patient according to the Bids format.
        Session is assumed to be 'one' for now.
        """
        make_bids_folders(subject=self.patientid, kind=self.modality, session='one',
            output_path = self.datafir, make_dir=True)

    def convert_to_bids(self, data, savePath, excluded_contacts=None):
        """
        Helper function to convert an edf file into bids format and create all 
        necessary metadata files corresponding to this data from Bids format.

        Parameters:
        -----------
        data: os.PathLike
            The path of the edf file to be loaded
        savePath: os.PathLike
            The path to save the resulting edf file. Named with Bids convention
        exluded_contacts: List[str]
            The list of contacts to exclude from conversion
        """
        if excluded_contacts is None:
            excluded_contacts = ["-", ""]
        raw = io.read_raw_edf(
            data,
            preload=False,
            verbose="ERROR",
            exclude=excluded_contacts,
            eog=None,
            misc=None,
        )
        outfile = savePath.split('_eeg')
        outfile = outfile[0]
        write_raw_bids(raw, outfile, output_path = self.datadir)

    def make_preprocessed_folder(self):
        """
        Creates the preproceesed data folder in the correct location if it does
        not already exist. Assumes session is 'one'
        """
        basedir = os.path.join(self.datadir, 'sub-', + self.patientid, 'ses-one')
        preprocessed_dir = os.path.join(basedir, 'preprocessed')
        if os.path.exists(preprocessed_dir):
            return 1
        else
            os.mkdir(preprocessed_dir)

    def make_results_folder(self):
        """
        Creates the results folder in the correct location if it does not already
        exist. Assumes session is 'one'
        """
        basedir = os.path.join(self.datadir, 'sub-' + self.patientid, 'ses-one')
        results_dir = os.path.join(basedir, 'results')
        if os.path.exists(results_dir):
            return 1
        else:
            os.mkdir(results_dir)

