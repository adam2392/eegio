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

    @property
    def numdatasets(self):
        return len(self.datasetsids)
    
    @property
    def numfifs(self):
        return len(self.fifids)

    def get_filesizes(self):
        """
        Get the total filesize of all the files attached for this patient.

        """
        size = 0
        for f in self.get_all_filepaths():
            size += os.path.getsize(f)
        return size

    def summary(self):
        basepath = os.path.join(self.datadir, self.patientid)
        print_dir_tree(basepath)

    def pickle_results(self):
        patient = self
        patient_pickle = os.path.join(self.datadir, 'sub-'+self.patientid, 'patient_object.obj')
        file_patient = open(patient_pickle, 'wb+')
        pickle.dump(patient, file_patient)


    def make_bids_folder_struct(self):
        #outpath = os.path.join(self.datadir, self.patientid)
        make_bids_folders(subject=self.patientid, kind=self.modality, session='one',
            output_path = self.datadir, make_dir=True)
    
    def load_metadata(self, metadata: dict):
        self.metadata = metadata

    def add_bad_channels(self, bad_channels: List[str]):
        self.bad_channels = bad_channels

    def add_wm_channels(self, wm_channels: List[str]):
        self.wm_channels = wm_channels

    def add_datasets_to_patient(self, datasetids: List[str]):
        numtasks = self.numdatasets
        basedir = os.path.join(self.datadir, 'sub-'+self.patientid, 'ses-one')
        for data in datasetids:
            numtasks += 1
            #task_num = f"{%s:02d}" % numtasks
            task_num = '01'
            basename = "sub-%s_ses-%s_task-%s_eeg.edf" % (self.patientid, "one", task_num)
            filename = os.pa
            th.join(basedir, 'eeg', basename)
            self.datasetids.extend(filename)
            self.convert_to_bids(data, filename)

    def add_fif_to_patient(self, fifids: List[str]):
        numtasks = self.numfifs
        if numtasks==0:
            self.make_preprocessed_folder()
        session = 'one'
        basedir = os.path.join(self.datadir, 'sub-'+self.patientid, "ses-"+session)
        for fif in fifids:
            numtasks += 1
            #task_num = f"{%s:02d}" % numtasks
            task_num = '01'
            basename = "sub-%s_ses-%s_task-%s_raw.fif" % (self.patientid, session, task_num)
            filename = os.path.join(basedir, 'preprocessed', basename)
            sh.move(fif, filename)
            self.fifids.extend(filename)

    def convert_to_bids(self, data, filename):
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
        basedir = os.path.join(self.datadir, 'sub-' + self.patientid, 'ses-one')
        preprocessed_dir = os.path.join(basedir, 'preprocessed')
        if os.path.exists(preprocessed_dir):
            return 1
        else:
            os.mkdir(preprocessed_dir)

    def make_results_folder(self):
        basedir = os.path.join(self.datadir, 'sub-' + self.patientid, 'ses-one')
        results_dir = os.path.join(basedir, 'results')
        if os.path.exists(results_dir):
            return 1
        else:
            os.mkdir(preprocessed_dir)

