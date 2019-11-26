import os
import pickle
import warnings
from typing import Union

import mne_bids
from mne_bids import make_bids_basename

from eegio.base.objects.bids.basebids import BaseBids
from eegio.base.objects.bids.bids_run import BidsRun
from eegio.base.utils.bids_helper import BidsBuilder
from eegio.loaders.bidsio import BidsWriter, BidsLoader


class BidsPatient(BaseBids):
    """
    The class for our patient data that inherits basic dataset functionality and follows Bids
    format. The additional components for this are related to the fact that each Patient can
    have multiple EEGTimeSeries, or Results related to it.

    Attributes
    ----------

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
            modality: str = "ieeg",
            metadata: dict = None,
            scans: dict = None,
            original_subject_id: str = None,
    ):
        super(BidsPatient, self).__init__(datadir=datadir, modality=modality)


        if scans is None:
            scans = {}

        self.subject_id = subject_id
        self.session_id = session_id
        self.managing_user = managing_user
        self.subjdir = os.path.join(self.datadir, 'sub-'+self.subject_id)
        self.metadata = metadata
        self.scans = scans

        new_subject = False

        if not os.path.exists(self.subjdir):
            warnings.warn(f"Subject directory does not exist at: {self.subjdir} "
                          f"We are going to create it for you!")
            # create bids folder for this subject
            BidsBuilder.make_bids_folder_struct(self.datadir, self.subject_id, self.modality, self.session_id)
            new_subject = True

        bids_basename = make_bids_basename(subject=self.subject_id, session=self.session_id)
        self.bids_basename = bids_basename

        # run to cache and search to get all datasets available for this patient
        self._get_all_datasets()

        # store parameters
        self._store_params()

        # create derivative folders
        self._create_derivative_folders()

        # instantiate a loader/writer
        self.loader = BidsLoader(bidsdir=datadir, subject_id=subject_id, session_id=session_id)
        self.writer = BidsWriter(datadir, subject_id, session_id)

        if new_subject:
            # Need to have subject in file in order to load and modify any metadata
            BidsBuilder.add_subject_to_participants(self.subject_id, self.loader, self.writer)
            # TODO: How do we add subject age/sex at this point

        # automatically update patient metadata
        self.resected_contacts = None
        self.original_subject_id = original_subject_id

        metadata = self._create_metadata()
        self._load_metadata(metadata)

        if self.numdatasets > 0:
            scans = self._create_scan_data()
            self._load_scans(scans)

    def __str__(self):
        return "{} Patient with {} Dataset Recordings ".format(
            self.subject_id, self.numdatasets
        )

    def _create_derivative_folders(self, types="all"):
        if types not in ["all", "preprocess", "results"]:
            raise ValueError("Can only create two types of derivative folders right now.")

        # create preprocess, results folders
        BidsBuilder.make_preprocessed_folder(self.datadir, self.subject_id, self.session_id)
        BidsBuilder.make_results_folder(self.datadir, self.subject_id, self.session_id)

    def get_subject_sessions(self):
        return mne_bids.utils.get_entity_vals(self.subjdir, 'ses')

    @property
    def numdatasets(self):
        # should be return len(self.dataset_fpaths). Fix updating _get_all_datasets
        return len(self.scans)

    @property
    def numfifs(self):
        return len([x for x in self.dataset_fpaths if x.endswith(".fif")])

    @property
    def numedfs(self):
        return len([x for x in self.edf_fpaths if x.endswith(".edf")])

    def _get_all_datasets(self):
        # Should update after creating a run, but does not work currently
        edffiles = self.query.get(extension='.edf', return_type='file')
        fiffiles = self.query.get(extension='.fif', return_type='file')
        self.edf_fpaths = edffiles
        self.dataset_fpaths = fiffiles

    def create_fake_example(self):
        patient_id = self.subject_id
        session_id = self.session_id
        original_subject_id = self.original_subject_id
        datadir = self.datadir
        patient = BidsPatient(patient_id, session_id, datadir, original_subject_id=original_subject_id)
        return patient

    def preprocess_edf_files(self, filelist: list = None, line_noise=60):
        """

        Parameters
        ----------
        filelist : list[os.PathLike]
            The list of all files to be converted into .fif format. Default is all edf files available
        line_noise : int
            The line noise to be eliminated. Typically 60 Hz in America

        Returns
        -------

        """
        if filelist is None:
            filelist = self._find_non_processed_files()
        for i, fpath in enumerate(filelist):
            if not os.path.exists(fpath):
                raise OSError(f"{fpath} doesn't exist. Please pass in valid filepaths.")
            if line_noise != 50:
                sidecardict = self.loader.load_sidecar_json()
                sidecardict["PowerLineFrequency"] = line_noise
                self.writer.write_sidecar_json()

            # TODO: Check if .fif file already exists. Otherwise convert to fif

    def _find_non_processed_files(self):
        """
        Finds the .edf files that have not been preprocessed into .fif files
        Returns
        -------

        """
        scans_df = self.loader.load_scans_tsv()
        non_processed = []
        colnames = scans_df.columns
        if 'original_filename' not in colnames:
            return scans_df['filename']
        for ind, row in scans_df.iterrows():
            if row['original_filename'] == '':
                non_processed.append(row['filename'])
        return non_processed

    def _create_run(self, scan):
        run_id = str(self.numdatasets).zfill(2)
        bids_basename = make_bids_basename(subject=self.subject_id, session=self.session_id, run=run_id)
        builder = BidsBuilder()
        builder.convert_to_bids(edf_fpath=scan, bids_dir=self.datadir, bids_basename=bids_basename)
        bidsrun = BidsRun(subject_id=self.subject_id, session_id=self.session_id, run_id=run_id, datadir=self.datadir,
                          modality=self.modality)
        builder.create_channel_info(bidsrun)
        scans = self._create_scan_data()
        self._load_scans(scans)
        self._get_all_datasets()
        return bidsrun

    def add_scans(self, scans: list):
        """
        Adds scans (runs) to the patient
        Parameters
        ----------
        scans : list[os.PathLike]

        Returns
        -------

        """
        for scan in scans:
            if not os.path.exists(scan):
                raise OSError(f'{scan} does not exist. Please pass existing scans')
            self._create_run(scan)

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

    def _load_scans(self, scans: dict):
        self.scans = scans

    def _create_scan_data(self):
        """
        Creates the scans dictionary from whatever is in the scans.tsv file

        Returns
        -------

        """
        scans_df = self.loader.load_scans_tsv()
        scans_dict = {}
        for ind, row in scans_df.iterrows():
            scans_dict[row['filename']] = {}
            for index, value in row.iteritems():
                scans_dict[row['filename']][index] = value
        return scans_dict

    def _load_metadata(self, metadata: dict):
        self.metadata = metadata


    def _create_metadata(self):
        """
        Creates the metadata dictionary from whatever is in the participants.tsv file
        Returns
        -------

        """
        participants_df = self.loader.load_participants_tsv()
        participants_series = participants_df.loc[participants_df['participant_id'] == self.subject_id].squeeze()
        series_dict = {'participant_id': participants_series.name}
        for index, value in participants_series.iteritems():
            series_dict[index] = value
        return series_dict

    def add_participants_field(self, column_id, description: Union[str, dict], subject_val: str = 'n/a',
                               default_val: str = 'n/a'):
        """

        Parameters
        ----------
        column_id : str
            The new column to add to the participants.tsv file
        description: Union[str, dict]
            The new column's description to be added to
        subject_val : str
            The value of column_id for this BidsPatient
        default_val :
            The value of column_id to fill for the rest of subjects in participants.tsv

        Returns
        -------

        """
        participants_dict = self.loader.load_participants_json()
        participants_df = self.loader.load_participants_tsv()
        participants_df = participants_df.set_index('participant_id')
        if column_id not in participants_dict.keys():
            participants_dict = BidsBuilder.add_field_to_dict(participants_dict, column_id, description)
            self.writer.write_participants_json(participants_dict)

        colnames = participants_df.columns
        if column_id not in colnames:
            add_list = [default_val]*participants_df.shape[0]
            participants_df[column_id] = add_list
            if subject_val != default_val:
                participants_df.at[self.subject_id, column_id] = subject_val
            self.writer.write_participants_tsv(participants_df)
        else:
            warnings.warn('Field already exists for the participants file. Modifying instead')
            self.modify_participants_file(column_id, subject_val)
        metadata = self._create_metadata()
        self._load_metadata(metadata)

    def remove_participants_field(self, column_id: str):
        """

        Parameters
        ----------
        column_id : str
            The name of the column to remove from the json and tsv files
        Returns
        -------

        """
        participants_dict = self.loader.load_participants_json()
        participants_df = self.loader.load_participants_tsv()
        if column_id not in participants_dict.keys():
            raise LookupError(f'The field {column_id} does not exist in the participants file')
        del participants_dict[column_id]
        self.writer.write_participants_json(participants_dict)
        colnames = participants_df.columns
        if column_id not in colnames:
            raise ValueError(f'The field {column_id} does not exist in the participants file')
        participants_df = participants_df.drop(column_id, axis=1)
        self.writer.write_participants_tsv(participants_df)
        metadata = self._create_metadata()
        self._load_metadata(metadata)

    def modify_participants_file(self, column_id: str, new_value: str):
        """
        Modify the BIDS dataset participants tsv file to include a new value

        Parameters
        ----------
        column_id : str
            The name of the column to which data should be modified. Must exist in the tsv file
        new_value : str
            The new value of the column_id for this BidsPatient
        Returns
        -------

        """
        participants_df = self.loader.load_participants_tsv()
        colnames = participants_df.columns
        if column_id not in colnames:
            raise ValueError(f'{column_id} not in the existing participants file. Add it first.')
        participants_df.at[self.subject_id, column_id] = new_value
        self.writer.write_participants_tsv(participants_df)
        metadata = self._create_metadata()
        self._load_metadata(metadata)

    def add_resected_contacts(self, resected_list):
        resected_str = " ".join(resected_list)
        self.add_participants_field('resected_contacts', 'The list of contacts corresponding to resected regions',
                                    resected_str, 'n/a')
        self.resected_contacts = resected_list

    def add_original_subject_id(self, original_subject_id):
        self.add_participants_field('original_subjectid', 'The subjectid in the clinical data source',
                                    original_subject_id, 'n/a')
        self.original_subject_id = original_subject_id
