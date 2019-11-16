import os
import pickle
import shutil as sh

from mne_bids import make_bids_basename

from typing import List

from eegio.base.config import EEGIO_DIR
from eegio.base.objects.bids.basebids import BaseBids


class BidsPatient(BaseBids):
    """
    The class for our patient data that inherits basic dataset functionality and follows Bids
    format. The additional components for this are related to the fact that each Patient can
    have multiple EEGTimeSeries, or Results related to it.

    Attributes
    ----------
    patientid: str
        patient identifier for this BidsPatient
    datadir: os.PathLike
        The base directory for storing BidsPatient data
    managing_user: str
        managing user that is tied to this BidsPatient. E.g. a clinical center, specific researcher, or specific doctor
    datasetids: List[str]
        List of filepaths of recording snapshots tied to this BidsPatient.
    fifids: List[str]
        List of filepaths of preprocessed recording snapshots tied to this BidsPatient.
    modality: str
        Modality of recording done to this patient. E.g. ecog, seeg, scalp, ieeg.
    samplerate: float
        sampling rate of recordings. E.g. 1000 Hz
    wm_channels: List[str]
        List of recording channels placed on white matter.
    bad_channels: List[str]
        List of recording channels not recording relevant signal. Excluded from analysis.
    metadata:
        TODO
    rz_contacts: List[str]
        List of annotated contacts included in the surgically resected zone.
    soz_contacts: List[str]
        List of annotated contacts thought to be the seizure onset zone.
    spread_contacts:
        TODO
    brain_region_hypo:
        TODO
    resected_region:
        TODO
    lesion_present: bool
        Whether the patient has a lesion present
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
        # center: str=None,
        # datasetids=None,
        # fifids=None,
        modality: str = "ieeg",
        # samplerate: float = None,
        # wm_channels=None,
        # bad_channels=None,
        # metadata=None,
        # rz_contacts=None,
        # soz_contacts=None,
        # spread_contacts=None,
        # brain_region_hypo: str = None,
        # resected_region: str = None,
        # lesion_present: bool = False,
    ):
        if datadir is None:
            datadir = os.path.join(EEGIO_DIR, 'BIDS_Subjects')
        
        super(BidsPatient, self).__init__(datadir=datadir, modality=modality)
        
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
        
        self.managing_user = managing_user
        self.center = center
        self.datasetids = datasetids
        self.fifids = fifids
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

        self.pickle_results()

        # store parameters
        self._store_params()

        if not os.path.exists(os.path.join(self.datadir)):
            self.make_bids_folder_struct()
            bids_basename = make_bids_basename(subject=patientid, session='one')
            self.bids_basename = bids_basename

    def __str__(self):
        return "{} Patient with {} Dataset Recordings " "totaling {} seconds".format(
            self.patientid, self.numdatasets, self.total_length_seconds
        )

    def _store_params(self):
        self.parameter_dict.update(
            managing_user=self.managing_user,
            center=self.center,
            datasetids=self.datasetids,
            fifids=self.fifids,
            samplerate=self.samplerate,
            wm_channels=self.wm_channels,
            bad_channels=self.bad_channels,
            metadata=self.metadata,
            rz_contacts=self.rz_contacts,
            soz_contacts=self.soz_contact
            spread_contacts=self.spread_contacts,
            brain_region_hypo=self.brain_region_hypo,
            resected_region=self.resected_region,
            lesion_present=self.legion_present
        )

    def get_subject_sessions(self):
        return mne_bids.utils.get_entity_vals(self.patdir, 'ses')
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
        """

        Returns
        -------
            Number of datasets in the patient
        """
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


    def pickle_results(self):
        """
        Save the BidsPatient object via pickling. Stored at the subject level.
        """
        patient = self
        patient_pickle = os.path.join(self.datadir, 'sub-'+self.patientid, 'patient_object.obj')
        file_patient = open(patient_pickle, 'wb+')
        pickle.dump(patient, file_patient)


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
        self._store_params()

    def add_wm_channels(self, wm_channels: List[str]):
        """
        Helper setter function to load additional wm_channels

        Parameters
        ----------
        wm_channels: List[str]
            The list of new white matter channels
        """
        self.wm_channels = wm_channels
        self._store_params()

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
        self._store_params()

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
        self._store_params()

    def load_dataset(self, dataset_num):
        """
        Loads a specified edf file into an EEGTimeSeries object.

        Parameters
        ----------
        dataset_num: int
            The number of the dataset in the BidsPatient's datasetids to be loaded

        Returns
        -------
        An EEGTimeSeries object containing the eeg data
        """
        # TODO: Adam
        #return eegts
    
    def load_fif(self, fif_num):
        """
        Loads a specified fif file into an EEGTimeSeries object.

        Parameters
        ----------
        fif_num: int
            The number of the fif file in the BidsPatient's fifids to be loaded

        Returns
        -------
        An EEGTimeSeries object containing the eeg data
        """
        # TODO: Adam
        #return eegts

    def add_to_participants(self, formatted_sheet_path):
        """
        Adds the data from the input formatted sheet and adds it to the Bids format
        participants.tsv file

        Parameters
        ----------
        formatted_sheet_path: os.PathLike
            The location of the formatted sheet to add

        """
        # TODO: Patrick
        pass

    def update_subject_channels(self):
        """
        The bids format has a tsv file per run to store all electrode channel information. 
        This function updates that file with the correct information about the signal type
        at each channel (e.g. eeg, eog, ecg...) and the status of each channel (e.g. good,
        wm, bad...)
        """
        basedir = os.path.join(self.datadir, 'sub-'+self.patientid, 'ses-one')
        eegdir = os.path.join(basedir,'eeg')
        channelFiles = []
        for file in os.listdir(eegdir):
            if file.endswith("channels.tsv"):
                channelFiles.append(os.path.join(eegdir, file))
        for f in channelFiles:
            # TODO: Patrick

    def update_subject_json(self):
        """
        The bids format has a json file per run to store all metadata information. 
        This function updates that file with the correct information about the recording
        """
        basedir = os.path.join(self.datadir, 'sub-'+self.patientid, 'ses-one')
        eegdir = os.path.join(basedir,'eeg')
        jsonFiles = []
        for file in os.listdir(eegdir):
            if file.endswith("eeg.json"):
                jsonFiles.append(os.path.join(eegdir, file))
        for f in jsonFiles:
            # TODO: Patrick

