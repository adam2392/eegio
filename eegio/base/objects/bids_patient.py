import os
import sys
import warnings
import pickle
import shutil as sh

import mne
from mne import io
from mne_bids import write_raw_bids, make_bids_basename, make_bids_folders
from mne_bids.utils import print_dir_tree

from typing import List, Dict

parentdir = os.path.join(os.getcwd(),'..')
sys.path.append(parentdir)
from config import EEGIO_DIR



class BidsPatient(object):
    """
    The class for our patient data that inherits basic dataset functionality and follows Bids
    format. The additional components for this are related to the fact that each Patient can
    have multiple EEGTimeSeries, or Results related to it.

    Attributes
    ----------
        :param patientid: patient identifier for this BidsPatient
        :type patientid: str
        :param datadir: The base directory for storing BidsPatient data
        :type datadir: os.PathLike
        :param managing_user: managing user that is tied to this BidsPatient. E.g. a clinical center, specific researcher, or specific doctor
        :type managing_user: str
        :param datasetids: List of filepaths of recording snapshots tied to this BidsPatient.
        :type datasetids: List[str]
        :param fifids: List of filepaths of preprocessed recording snapshots tied to this BidsPatient.
        :type fifids: List[str]
        :param modality: modality of recording done to this patient. E.g. ecog, seeg, scalp, ieeg.
        :type modality: str
        :param samplerate: sampling rate for recordings. E.g. 1000 Hz
        :type samplerate: float
        :param wm_channels: List of recording channels placed on white matter
        :type wm_channels: List[str]
        :param bad_channels: List of recording channels not recording relevant signal.
        :type bad_channels: List[str]
        :param metadata:
        :type metadata:
        :param rz_contacts: List of annotated contacts included in the surgically resected zone.
        :type rz_contacts: List[str]
        :param soz_contacts: List of annotated contacts thought to be the seizure onset zone.
        :type soz_contacts: List[str]
        :param spread_contacts: 
        :type spread_contacts:
        :param brain_region_hypo:
        :type brain_region_hypo:
        :param resected_region:
        :type resected_region:
        :param lesion_present:
        :type lesion_present:

    Notes
    -----

    Examples
    --------
    """

    def __init__(
        self,
        patientid: str,
        datadir: os.PathLike=None,
        managing_user: str=None,
        center: str=None,
        datasetids=None,
        fifids=None,
        modality: str = "ieeg",
        samplerate: float = None,
        wm_channels=None,
        bad_channels=None,
        metadata=None,
        rz_contacts=None,
        soz_contacts=None,
        spread_contacts=None,
        brain_region_hypo: str = None,
        resected_region: str = None,
        lesion_present: bool = False,
    ):
        if spread_contacts is None:
            spread_contacts = []
        if soz_contacts is None:
            soz_contacts = []
        if rz_contacts is None:
            rz_contacts = []
        if bad_channels is None:
            bad_channels = []
        if metadata is None:
            metadata = dict()
        if wm_channels is None:
            wm_channels  = []
        if datasetids is None:
            datasetids = []
        if fifids is None:
            fifids = []
        if datadir is None:
            datadir = os.path.join(EEGIO_DIR, 'BIDS_Subjects')
        
        self.patientid = patientid
        self.datadir = datadir
        self.managing_user = managing_user
        self.center = center
        self.datasetids = datasetids
        self.fifids = fifids
        self.modality = modality
        self.samplerate = samplerate

        self.wm_channels = wm_channels
        self.bad_channels = bad_channels
        
        self.metadata = metadata

        self.rz_contacts = rz_contacts
        self.soz_contacts = soz_contacts
        self.spread_contacts = spread_contacts
        self.brain_region_hypo = brain_region_hypo
        self.resected_region = resected_region
        self.lesion_present = lesion_present

        if (not os.path.exists(os.path.join(self.datadir, self.patientid))):
            self.make_bids_folder_struct()
            bids_basename = make_bids_basename(subject=patientid, session='one')
            self.bids_basename = bids_basename

        self.pickle_results()

    def __str__(self):
        return "{} Patient with {} Dataset Recordings " "totaling {} seconds".format(
            self.patientid, self.numdatasets, self.total_length_seconds
        )

    @classmethod
    def create_fake_example(self):
        import tempfile
        from eegio.base.objects.dataset.eegts_object import EEGTimeSeries

        eegts = EEGTimeSeries.create_fake_example()
        patientid = "testid"
        with tempfile.TemporaryDirectory() as datadir:
            patient = BidsPatient(patientid, datadir)
        return patient

    @property
    def numdatasets(self):
        return len(self.datasetsids)
    
    @property
    def numfifs(self):
        return len(self.fifids)

    def get_filesizes(self):
        """
        Get the total filesize of all the files attached for this patient.
        
        Returns
        -------
        int
            size of all files
        """
        size = 0
        for f in self.get_all_filepaths():
            size += os.path.getsize(f)
        return size

    def summary(self):
        """
        Print a summary of the BidsPatient in the form of a directory tree showing
        all available data
        """
        basepath = os.path.join(self.datadir, self.patientid)
        print_dir_tree(basepath)

    def pickle_results(self):
        """
        Save the BidsPatient object via pickling. Stored at the subject level.
        """
        patient = self
        patient_pickle = os.path.join(self.datadir, 'sub-'+self.patientid, 'patient_object.obj')
        file_patient = open(patient_pickle, 'wb+')
        pickle.dump(patient, file_patient)


    def make_bids_folder_struct(self):
        """
        Creates the empty folder structure for a new BidsPatient according to the Bids format.
        Session is assumed to be 'one' for now.
        """
        make_bids_folders(subject=self.patientid, kind=self.modality, session='one',
            output_path = self.datadir, make_dir=True)
    
    def load_metadata(self, metadata: dict):
        self.metadata = metadata

    def add_bad_channels(self, bad_channels: List[str]):
        """
        Helper setter function to load additional bad_channels

        Parameters
        ----------
        bad_channels: List[str]
            The list of new bad channels
        """
        self.bad_channels = bad_channels

    def add_wm_channels(self, wm_channels: List[str]):
        """
        Helper setter function to load additional wm_channels

        Parameters
        ----------
        wm_channels: List[str]
            The list of new white matter channels
        """
        self.wm_channels = wm_channels

    def add_datasets_to_patient(self, datasetids: List[str]):
        """
        Helper setter function to add additional dataset ids to this BidsPatient.
        Converts any passed datasets to Bids format.

        Parameters
        ----------
        datasetids: List[str]
            A list of string paths to be added to the patient
        """
        numtasks = self.numdatasets
        basedir = os.path.join(self.datadir, 'sub-'+self.patientid, 'ses-one')
        for data in datasetids:
            numtasks += 1
            #task_num = f"{%s:02d}" % numtasks
            task_num = '01'
            basename = "sub-%s_ses-%s_task-%s_eeg.edf" % (self.patientid, "one", task_num)
            filename = os.path.join(basedir, 'eeg', basename)
            self.datasetids.extend(filename)
            self.convert_to_bids(data, filename)

    def add_fif_to_patient(self, fifids: List[str]):
        """
        Helper setter function to add additional fif ids to this BidsPatient.
        Converts any passed fif files to Bids format.

        Parameters
        ----------
        fifids: List[str]
            A list of string paths of fif files to be added to the patient.
        """
        numtasks = self.numfifs
        if numtasks==0:
            self.make_preprocessed_folder()
        session = 'one'
        basedir = os.path.join(self.datadir, 'sub-'+self.patientid, "ses-"+session)
        for fif in fifids:
            numtasks += 1
            task_num = f"{%f:02d}" % numtasks
            basename = "sub-%s_ses-%s_task-%s_raw.fif" % (self.patientid, session, task_num)
            filename = os.path.join(basedir, 'preprocessed', basename)
            sh.move(fif, filename)
            self.fifids.extend(filename)

    def convert_to_bids(self, data, filename):
        """
        Helper function to convert an edf file into bids format and create all 
        necessary metadata files corresponding to this data from Bids format.

        Parameters
        ----------
        data: os.PathLike
            The path of the edf file to be loded
        filename: os.PathLike
            The path to save the resulting edf file. Named with Bids convention
        """
        excluded_contacts = ["-", ""]
        raw = io.read_raw_edf(
            data,
            preload=False,
            verbose="ERROR",
            exclude=excluded_contacts,
            eog=None,
            misc=None,
        )
        samplerate = raw.info["sfreq"]
        annotations = mne.events_from_annotations(
            raw, use_rounding=True, chunk_duration=None, verbose=False
        )
        outfile = filename.split('_eeg')
        outfile = outfile[0]
        write_raw_bids(raw, outfile, output_path = self.datadir)

    def make_preprocessed_folder(self):
        """
        Creates the preprocessed data folder in the correct location if it does
        not already exist. Assumes session is 'one'
        """
        basedir = os.path.join(self.datadir, 'sub-' + self.patientid, 'ses-one')
        preprocessed_dir = os.path.join(basedir, 'preprocessed')
        if os.path.exists(preprocessed_dir):
            return 1
        else:
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
            os.mkdir(preprocessed_dir)

