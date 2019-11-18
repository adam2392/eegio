import os
import pickle
import warnings
from typing import Union

import mne_bids
from mne_bids import make_bids_basename

from eegio.base.objects.bids.basebids import BaseBids
from eegio.base.utils.bids_helper import BidsBuilder
from eegio.loaders.bidsio import BidsWriter, BidsLoader


class BidsPatient(BaseBids):
    """
    The class for our patient data that inherits basic dataset functionality and follows Bids
    format. The additional components for this are related to the fact that each Patient can
    have multiple EEGTimeSeries, or Results related to it.

    Attributes
    ----------
    subject_id: str
        patient identifier for this BidsPatient
    session_id: str
        the session that the patient data belongs to
    datadir: os.PathLike
        The base directory for storing BidsPatient data
    managing_user: str
        managing user that is tied to this BidsPatient. E.g. a clinical center, specific researcher, or specific doctor
    original_subject_id: str
        the original subject_id in the data source
    modality: str
        Modality of recording done to this patient. E.g. ecog, seeg, scalp, ieeg.
    metadata:
        TODO

    Notes
    -----

    Examples
    --------
    """

    def __init__(
            self,
            subject_id: str,
            session_id: str = None,
            datadir: Union[str, os.PathLike] = None,
            managing_user: str = None,
            original_subject_id: str = None,
            modality: str = "ieeg",
            resected_contacts: list[str] = None,
    ):
        super(BidsPatient, self).__init__(datadir=datadir, modality=modality)

        self.subject_id = subject_id
        self.session_id = session_id
        self.managing_user = managing_user
        self.original_subject_id = original_subject_id
        self.resected_contacts = resected_contacts
        self.subjdir = os.path.join(self.datadir, self.subject_id)

        if not os.path.exists(self.subjdir):
            warnings.warn(f"Subject directory does not exist at: {self.subjdir} "
                          f"We are going to create it for you!")
            # create bids folder for this subject
            BidsBuilder.make_bids_folder_struct(self.datadir, self.subject_id, self.modality)
            bids_basename = make_bids_basename(subject=self.subject_id, session=self.session_id)
            self.bids_basename = bids_basename

        # run to cache and search to get all datasets available for this patient
        self._get_all_datasets()

        # store parameters
        self._store_params()

        # create derivative folders
        self._create_derivative_folders()

        # instantiate a loader/writer
        self.loader = BidsLoader(datadir, subject_id, session_id)
        self.writer = BidsWriter(datadir, subject_id, session_id)

        # automatically update patient metadata
        self._add_resected_contacts()
        self._add_original_patient_name()

    def __str__(self):
        return "{} Patient with {} Dataset Recordings ".format(
            self.subject_id, self.numdatasets
        )

    def _create_derivative_folders(self, types="all"):
        if types not in ["all", "preprocess", "results"]:
            raise ValueError("Can only create two types of derivative folders right now.")

        # create preprocess, results folders
        BidsBuilder.make_preprocessed_folder(self.datadir, self.subject_id)
        BidsBuilder.make_results_folder(self.datadir, self.subject_id)

    def get_subject_sessions(self):
        return mne_bids.utils.get_entity_vals(self.subjdir, 'ses')

    @property
    def numdatasets(self):
        return len(self.dataset_fpaths)

    @property
    def numfifs(self):
        return len([x for x in self.dataset_fpaths if x.endswith(".fif")])

    @property
    def numedfs(self):
        return len([x for x in self.edf_fpaths if x.endswith(".edf")])

    def _get_all_datasets(self):
        edffiles = self.query.get(extension='.edf', return_type='file')
        fiffiles = self.query.get(extension='.fif', return_type='file')
        self.edf_fpaths = edffiles
        self.dataset_fpaths = fiffiles

    @classmethod
    def create_fake_example(self):
        import tempfile
        patient_id = self.subject_id
        session_id = self.session_id
        original_subject_id = self.original_subject_id
        with tempfile.TemporaryDirectory() as datadir:
            patient = BidsPatient(patient_id, session_id, datadir, original_subject_id=original_subject_id)
        return patient

    def preprocess_edf_files(self, filelist, line_noise=60):
        output_fpaths = []

        # preprocess all edf files first
        for i, fpath in enumerate(filelist):
            if not os.path.exists(fpath):
                raise OSError(f"{fpath} doesn't exist. Please pass in valid filepaths.")

            bids_basename = make_bids_basename(self.subject_id, self.session_id, run=i)
            bids_edf_fpath = BidsBuilder.convert_to_bids(fpath, self.datadir, bids_basename, overwrite=True)

            # TODO: Helper functions here

            # overwrite default line frequency noise in MNEBids
            if line_noise != 50:
                sidecardict = self.loader.load_sidecar_json()
                sidecardict["PowerLineFrequency"] = line_noise
                self.writer.write_sidecar_json()

            output_fpaths.append(bids_edf_fpath)

        # process these all into fif files
        return output_fpaths

    def load_dataset(self, session_id, run_id, kind):
        """
        Loads in MNE.io.Raw dataset based on session id and run id passed in. Also returns a metadata structure

        Parameters
        ----------
        session_id :
        run_id :

        Returns
        -------

        """
        dataset = self.loader.load_dataset(session_id, run_id)
        metadata = self.loader.load_sidecar_json()

    def _add_resected_contacts(self):
        resected_list = " ".join(self.resected_contacts)

        participants_data = self.loader.load_participants_tsv()
        participants_json = self.loader.load_participants_json()
        if 'resected_contacts' not in participants_json.keys():
            participants_json = BidsBuilder.add_participant_field('resected_contacts', [], ['The list of contacts corresponding to '
                                                                        'resected regions'])
            self.writer.write_participants_json(participants_json)
        colnames = participants_data.columns
        if 'resected_contacts' not in colnames:
            resected = [''] * participants_data.shape[0]
            participants_data['resected_contacts'] = resected
        for ind, row in participants_data.iterrows():
            if row['participant_id'] == self.subject_id:
                participants_data['paricipant_id'][ind] = resected_list
        self.writer.write_participants_tsv(participants_data)

    def _add_original_patient_name(self):
        participants_data = self.loader.load_participants_tsv()
        participants_json = self.loader.load_participants_json()
        if 'original_subjectid' not in participants_json.keys():
            participants_json = BidsBuilder.add_participant_field(participants_json, 'original_subjectid', [],
                                              ['The subjectid in the clinical data source'])
            self.writer.write_participants_json(participants_json)
        colnames = participants_data.columns
        if 'original_subjectid' not in colnames:
            originals = [''] * participants_data.shape[0]
            participants_data['original'] = originals
        for ind, row in participants_data.iterrows():
            if row['participant_id'] == self.subject_id:
                participants_data['participant_id'][ind] = self.original_subject_id
        self.writer.write_participants_tsv(participants_data)
        # TODO: Add the self.original_subject_id field into the DataFrame and write to tsv


    def get_filesizes(self):
        """
        Get the total filesize of all the files attached for this patient.

        Returns
        -------
        int
            size of all files
        """
        size = 0
        for f in self.dataset_fpaths():
            size += os.path.getsize(f)
        return size

    def pickle_results(self):
        """
        Save the BidsPatient object via pickling. Stored at the subject level.
        """
        patient = self
        patient_pickle = os.path.join(self.datadir, 'sub-' + self.patientid, 'patient_object.obj')
        file_patient = open(patient_pickle, 'wb+')
        pickle.dump(patient, file_patient)

    def load_metadata(self, metadata: dict):
        self.metadata = metadata


    # def add_bad_channels(self, bad_channels: List[str]):
    #     """
    #     Helper setter function to load additional bad_channels
    #
    #     Parameters
    #     ----------
    #     bad_channels: List[str]
    #         The list of new bad channels
    #     """
    #     self.bad_channels = bad_channels
    #     self._store_params()
    #
    # def add_wm_channels(self, wm_channels: List[str]):
    #     """
    #     Helper setter function to load additional wm_channels
    #
    #     Parameters
    #     ----------
    #     wm_channels: List[str]
    #         The list of new white matter channels
    #     """
    #     self.wm_channels = wm_channels
    #     self._store_params()
    #
    # def add_datasets_to_patient(self, datasetids: List[str]):
    #     """
    #     Helper setter function to add additional dataset ids to this BidsPatient.
    #     Converts any passed datasets to Bids format.
    #
    #     Parameters
    #     ----------
    #     datasetids: List[str]
    #         A list of string paths to be added to the patient
    #     """
    #     numtasks = self.numdatasets
    #     basedir = os.path.join(self.datadir, 'sub-' + self.patientid, 'ses-one')
    #     for data in datasetids:
    #         numtasks += 1
    #         # task_num = f"{%s:02d}" % numtasks
    #         task_num = '01'
    #         basename = "sub-%s_ses-%s_task-%s_eeg.edf" % (self.patientid, "one", task_num)
    #         filename = os.path.join(basedir, 'eeg', basename)
    #         self.datasetids.extend(filename)
    #         self.convert_to_bids(data, filename)
    #     self._store_params()
    #
    # def add_fif_to_patient(self, fifids: List[str]):
    #     """
    #     Helper setter function to add additional fif ids to this BidsPatient.
    #     Converts any passed fif files to Bids format.
    #
    #     Parameters
    #     ----------
    #     fifids: List[str]
    #         A list of string paths of fif files to be added to the patient.
    #     """
    #     numtasks = self.numfifs
    #     if numtasks == 0:
    #         self.make_preprocessed_folder()
    #     session = 'one'
    #     basedir = os.path.join(self.datadir, 'sub-' + self.patientid, "ses-" + session)
    #     for fif in fifids:
    #         numtasks += 1
    #         task_num = f"{% f:02d}" % numtasks
    #         basename = "sub-%s_ses-%s_task-%s_raw.fif" % (self.patientid, session, task_num)
    #         filename = os.path.join(basedir, 'preprocessed', basename)
    #         sh.move(fif, filename)
    #         self.fifids.extend(filename)
    #     self._store_params()
    #
    # def load_dataset(self, dataset_num):
    #     """
    #     Loads a specified edf file into an EEGTimeSeries object.
    #
    #     Parameters
    #     ----------
    #     dataset_num: int
    #         The number of the dataset in the BidsPatient's datasetids to be loaded
    #
    #     Returns
    #     -------
    #     An EEGTimeSeries object containing the eeg data
    #     """
    #     # TODO: Adam
    #     # return eegts
    #
    # def load_fif(self, fif_num):
    #     """
    #     Loads a specified fif file into an EEGTimeSeries object.
    #
    #     Parameters
    #     ----------
    #     fif_num: int
    #         The number of the fif file in the BidsPatient's fifids to be loaded
    #
    #     Returns
    #     -------
    #     An EEGTimeSeries object containing the eeg data
    #     """
    #     # TODO: Adam
    #     # return eegts
    #
    # def add_to_participants(self, formatted_sheet_path):
    #     """
    #     Adds the data from the input formatted sheet and adds it to the Bids format
    #     participants.tsv file
    #
    #     Parameters
    #     ----------
    #     formatted_sheet_path: os.PathLike
    #         The location of the formatted sheet to add
    #
    #     """
    #     # TODO: Patrick
    #     pass
    #
    # def update_subject_channels(self):
    #     """
    #     The bids format has a tsv file per run to store all electrode channel information.
    #     This function updates that file with the correct information about the signal type
    #     at each channel (e.g. eeg, eog, ecg...) and the status of each channel (e.g. good,
    #     wm, bad...)
    #     """
    #     basedir = os.path.join(self.datadir, 'sub-' + self.patientid, 'ses-one')
    #     eegdir = os.path.join(basedir, 'eeg')
    #     channelFiles = []
    #     for file in os.listdir(eegdir):
    #         if file.endswith("channels.tsv"):
    #             channelFiles.append(os.path.join(eegdir, file))
    #     for f in channelFiles:
    #
    # # TODO: Patrick
    #
    # def update_subject_json(self):
    #     """
    #     The bids format has a json file per run to store all metadata information.
    #     This function updates that file with the correct information about the recording
    #     """
    #     basedir = os.path.join(self.datadir, 'sub-' + self.patientid, 'ses-one')
    #     eegdir = os.path.join(basedir, 'eeg')
    #     jsonFiles = []
    #     for file in os.listdir(eegdir):
    #         if file.endswith("eeg.json"):
    #             jsonFiles.append(os.path.join(eegdir, file))
    #     for f in jsonFiles:
    # # TODO: Patrick
